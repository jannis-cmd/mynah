package app

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/ErniConcepts/mynah/internal/llm"
	"github.com/ErniConcepts/mynah/internal/memory"
	"github.com/ErniConcepts/mynah/internal/storage"
)

type Config struct {
	DataDir          string
	OpenAIAPIKey     string
	OpenAIBaseURL    string
	OpenAIModel      string
	MemoryCharLimit  int
	ProfileCharLimit int
	RecallLimit      int
	Debug            bool
	DebugWriter      io.Writer
}

type Service struct {
	cfg       Config
	llmClient llmClient
}

type llmClient interface {
	GenerateReply(ctx context.Context, request llm.ReplyRequest) (string, error)
	ReviseMemory(ctx context.Context, request llm.MemoryRevisionRequest) (llm.MemoryRevision, error)
}

type InspectResult struct {
	TenantID         string
	AgentID          string
	UserID           string
	AgentRoot        string
	MemoryDoc        string
	ProfileDoc       string
	UserDoc          string
	MemoryProvenance storage.RevisionProvenance
	UserProvenance   storage.RevisionProvenance
	RejectedRevision storage.RejectedRevision
	RecentMessages   []storage.Message
}

type EvalCase struct {
	Name           string   `json:"name"`
	Messages       []string `json:"messages"`
	ExpectContains []string `json:"expect_contains"`
}

type EvalCaseResult struct {
	Name           string        `json:"name"`
	Passed         bool          `json:"passed"`
	Duration       time.Duration `json:"duration"`
	LastReply      string        `json:"last_reply"`
	FailureReason  string        `json:"failure_reason,omitempty"`
	ExpectContains []string      `json:"expect_contains,omitempty"`
}

type EvalRunResult struct {
	StartedAt  time.Time        `json:"started_at"`
	FinishedAt time.Time        `json:"finished_at"`
	Duration   time.Duration    `json:"duration"`
	Cases      []EvalCaseResult `json:"cases"`
	Passed     int              `json:"passed"`
	Failed     int              `json:"failed"`
}

func New(cfg Config) (*Service, error) {
	client, err := llm.NewClient(llm.Config{
		APIKey:  cfg.OpenAIAPIKey,
		BaseURL: cfg.OpenAIBaseURL,
		Model:   cfg.OpenAIModel,
	})
	if err != nil {
		return nil, err
	}

	return &Service{
		cfg:       cfg,
		llmClient: client,
	}, nil
}

func (s *Service) InitAgent(tenantID, agentID string) error {
	paths := storage.NewAgentPaths(s.cfg.DataDir, tenantID, agentID)
	if err := storage.EnsureAgentPaths(paths); err != nil {
		return err
	}

	store, err := storage.NewSessionStore(paths.HistoryPath)
	if err != nil {
		return err
	}
	defer store.Close()

	return store.EnsureSchema()
}

func (s *Service) ChatOnce(ctx context.Context, tenantID, agentID, userID, sessionID, userInput string) (string, error) {
	if strings.TrimSpace(userID) == "" {
		return "", fmt.Errorf("user_id is required")
	}

	paths := storage.NewAgentPaths(s.cfg.DataDir, tenantID, agentID)
	if err := storage.EnsureAgentPaths(paths); err != nil {
		return "", err
	}

	fileStore := storage.NewFileStore(paths, s.cfg.MemoryCharLimit, s.cfg.ProfileCharLimit)
	memoryDoc, err := fileStore.ReadMemory()
	if err != nil {
		return "", err
	}
	profileDoc, err := fileStore.ReadProfile()
	if err != nil {
		return "", err
	}
	userDoc, err := fileStore.ReadUserProfile(userID)
	if err != nil {
		return "", err
	}
	if err := memory.ValidateMemoryDocument(memoryDoc, s.cfg.MemoryCharLimit); err != nil && strings.TrimSpace(memoryDoc) != "" {
		s.debugf("stored_memory ignored reason=%q", err)
		memoryDoc = ""
	}
	if err := memory.ValidateProfileDocument(profileDoc, s.cfg.ProfileCharLimit); err != nil && strings.TrimSpace(profileDoc) != "" {
		s.debugf("stored_profile ignored reason=%q", err)
		profileDoc = ""
	}
	if err := memory.ValidateUserDocument(userDoc, s.cfg.ProfileCharLimit); err != nil && strings.TrimSpace(userDoc) != "" {
		s.debugf("stored_user ignored reason=%q", err)
		userDoc = ""
	}

	sessionStore, err := storage.NewSessionStore(paths.HistoryPath)
	if err != nil {
		return "", err
	}
	defer sessionStore.Close()

	if err := sessionStore.EnsureSchema(); err != nil {
		return "", err
	}
	if err := sessionStore.EnsureSessionForUser(sessionID, userID); err != nil {
		return "", err
	}
	if err := sessionStore.AppendMessage(sessionID, "user", userInput, time.Now().UTC()); err != nil {
		return "", err
	}
	s.debugf("chat_start tenant=%s agent=%s user=%s session=%s user_chars=%d", tenantID, agentID, userID, sessionID, len(userInput))

	recentMessages, err := sessionStore.RecentMessages(sessionID, 8)
	if err != nil {
		return "", err
	}

	recallContext, err := s.buildRecallContext(sessionStore, sessionID, userID, userInput)
	if err != nil {
		return "", err
	}
	s.debugf("prompt_context recent_messages=%d memory_chars=%d profile_chars=%d user_chars=%d recall_chars=%d", len(recentMessages), len(memoryDoc), len(profileDoc), len(userDoc), len(recallContext))

	reply, err := s.llmClient.GenerateReply(ctx, llm.ReplyRequest{
		AgentID:        agentID,
		MemoryDoc:      memoryDoc,
		ProfileDoc:     profileDoc,
		UserDoc:        userDoc,
		RecallContext:  recallContext,
		RecentMessages: toPromptMessages(recentMessages),
	})
	if err != nil {
		return "", err
	}

	if err := sessionStore.AppendMessage(sessionID, "assistant", reply, time.Now().UTC()); err != nil {
		return "", err
	}
	s.debugf("assistant_reply chars=%d", len(reply))

	revision, err := s.llmClient.ReviseMemory(ctx, llm.MemoryRevisionRequest{
		AgentID:       agentID,
		MemoryDoc:     memoryDoc,
		UserDoc:       userDoc,
		UserMessage:   userInput,
		AssistantText: reply,
		MemoryLimit:   s.cfg.MemoryCharLimit,
		UserLimit:     s.cfg.ProfileCharLimit,
	})
	if err != nil {
		s.debugf("memory_revision skipped error=%v", err)
		return reply, nil
	}
	s.debugf("memory_revision reason=%q operation_count=%d", revision.Reason, len(revision.Operations))

	writeRejectedRevision := func(rejectionErr error) {
		if rejectionErr == nil {
			return
		}
		_ = fileStore.WriteRejectedRevision(storage.RejectedRevision{
			Timestamp:      time.Now().UTC(),
			UserID:         userID,
			SessionID:      sessionID,
			Message:        strings.TrimSpace(userInput),
			Reason:         strings.TrimSpace(revision.Reason),
			RejectionError: rejectionErr.Error(),
			Operations:     revision.Operations,
		})
	}

	nextMemoryDoc, nextUserDoc, err := memory.ApplyMemoryOperations(memoryDoc, userDoc, userID, revision.Operations)
	if err != nil {
		s.debugf("memory_revision rejected error=%q", err)
		writeRejectedRevision(err)
		return reply, nil
	}
	memoryChanged := strings.TrimSpace(nextMemoryDoc) != strings.TrimSpace(memoryDoc)
	userChanged := strings.TrimSpace(nextUserDoc) != strings.TrimSpace(userDoc)

	provenance := storage.RevisionProvenance{
		UserID:    userID,
		SessionID: sessionID,
		Timestamp: time.Now().UTC(),
		Reason:    strings.TrimSpace(revision.Reason),
		Message:   strings.TrimSpace(userInput),
	}

	if err := memory.ValidateMemoryDocument(nextMemoryDoc, s.cfg.MemoryCharLimit); err != nil {
		s.debugf("memory_revision rejected reason=%q", err)
		writeRejectedRevision(err)
		return reply, nil
	}
	if err := memory.ValidateUserDocument(nextUserDoc, s.cfg.ProfileCharLimit); err != nil {
		s.debugf("user_revision rejected reason=%q", err)
		writeRejectedRevision(err)
		return reply, nil
	}

	if memoryChanged {
		if err := fileStore.WriteMemory(nextMemoryDoc); err != nil {
			return reply, err
		}
		if err := fileStore.WriteMemoryProvenance(provenance); err != nil {
			return reply, err
		}
	}
	if userChanged {
		if err := fileStore.WriteUserProfile(userID, nextUserDoc); err != nil {
			return reply, err
		}
		if err := fileStore.WriteUserProfileProvenance(userID, provenance); err != nil {
			return reply, err
		}
	}

	return reply, nil
}

func (s *Service) InspectAgent(tenantID, agentID, userID string, messageLimit int) (InspectResult, error) {
	paths := storage.NewAgentPaths(s.cfg.DataDir, tenantID, agentID)
	if err := storage.EnsureAgentPaths(paths); err != nil {
		return InspectResult{}, err
	}

	fileStore := storage.NewFileStore(paths, s.cfg.MemoryCharLimit, s.cfg.ProfileCharLimit)
	memoryDoc, err := fileStore.ReadMemory()
	if err != nil {
		return InspectResult{}, err
	}
	profileDoc, err := fileStore.ReadProfile()
	if err != nil {
		return InspectResult{}, err
	}
	memoryMeta, err := fileStore.ReadMemoryProvenance()
	if err != nil {
		return InspectResult{}, err
	}
	rejectedRevision, err := fileStore.ReadRejectedRevision()
	if err != nil {
		return InspectResult{}, err
	}
	var userDoc string
	var userMeta storage.RevisionProvenance
	if strings.TrimSpace(userID) != "" {
		userDoc, err = fileStore.ReadUserProfile(userID)
		if err != nil {
			return InspectResult{}, err
		}
		userMeta, err = fileStore.ReadUserProfileProvenance(userID)
		if err != nil {
			return InspectResult{}, err
		}
	}

	sessionStore, err := storage.NewSessionStore(paths.HistoryPath)
	if err != nil {
		return InspectResult{}, err
	}
	defer sessionStore.Close()

	if err := sessionStore.EnsureSchema(); err != nil {
		return InspectResult{}, err
	}
	messages, err := sessionStore.RecentMessagesAcrossAgent(messageLimit)
	if err != nil {
		return InspectResult{}, err
	}

	return InspectResult{
		TenantID:         tenantID,
		AgentID:          agentID,
		UserID:           userID,
		AgentRoot:        paths.RootPath,
		MemoryDoc:        memoryDoc,
		ProfileDoc:       profileDoc,
		UserDoc:          userDoc,
		MemoryProvenance: memoryMeta,
		UserProvenance:   userMeta,
		RejectedRevision: rejectedRevision,
		RecentMessages:   messages,
	}, nil
}

func (s *Service) RunEval(ctx context.Context, tenantID, agentID string, cases []EvalCase) (EvalRunResult, error) {
	started := time.Now()
	results := make([]EvalCaseResult, 0, len(cases))

	for index, evalCase := range cases {
		name := strings.TrimSpace(evalCase.Name)
		if name == "" {
			name = fmt.Sprintf("case_%02d", index+1)
		}
		caseStart := time.Now()
		sandboxDir, cleanup, err := s.prepareEvalSandbox(tenantID, agentID, name)
		if err != nil {
			return EvalRunResult{}, err
		}

		caseService, err := New(Config{
			DataDir:          sandboxDir,
			OpenAIAPIKey:     s.cfg.OpenAIAPIKey,
			OpenAIBaseURL:    s.cfg.OpenAIBaseURL,
			OpenAIModel:      s.cfg.OpenAIModel,
			MemoryCharLimit:  s.cfg.MemoryCharLimit,
			ProfileCharLimit: s.cfg.ProfileCharLimit,
			RecallLimit:      s.cfg.RecallLimit,
			Debug:            s.cfg.Debug,
			DebugWriter:      s.cfg.DebugWriter,
		})
		if err != nil {
			cleanup()
			return EvalRunResult{}, err
		}

		sessionID := NewSessionID()
		userID := "eval_user"
		var lastReply string
		var failure string

		for _, message := range evalCase.Messages {
			reply, err := caseService.ChatOnce(ctx, tenantID, agentID, userID, sessionID, message)
			if err != nil {
				failure = err.Error()
				break
			}
			lastReply = reply
		}

		passed := failure == ""
		if passed && len(evalCase.ExpectContains) > 0 {
			lowerReply := strings.ToLower(lastReply)
			for _, needle := range evalCase.ExpectContains {
				if !strings.Contains(lowerReply, strings.ToLower(needle)) {
					passed = false
					failure = fmt.Sprintf("reply missing expected substring %q", needle)
					break
				}
			}
		}

		results = append(results, EvalCaseResult{
			Name:           name,
			Passed:         passed,
			Duration:       time.Since(caseStart),
			LastReply:      lastReply,
			FailureReason:  failure,
			ExpectContains: slices.Clone(evalCase.ExpectContains),
		})

		cleanup()
	}

	finished := time.Now()
	run := EvalRunResult{
		StartedAt:  started,
		FinishedAt: finished,
		Duration:   finished.Sub(started),
		Cases:      results,
	}
	for _, result := range results {
		if result.Passed {
			run.Passed++
		} else {
			run.Failed++
		}
	}
	return run, nil
}

func LoadEvalCases(path string) ([]EvalCase, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var cases []EvalCase
	if err := json.Unmarshal(raw, &cases); err == nil {
		return cases, nil
	}

	var wrapped struct {
		Cases []EvalCase `json:"cases"`
	}
	if err := json.Unmarshal(raw, &wrapped); err != nil {
		return nil, err
	}
	return wrapped.Cases, nil
}

func (s *Service) buildRecallContext(store *storage.SessionStore, sessionID, userID, userInput string) (string, error) {
	sections := make([]string, 0, 2)
	lower := strings.ToLower(userInput)

	if strings.Contains(lower, "yesterday") {
		start, end := yesterdayBounds()
		messages, err := store.MessagesBetweenForUser(userID, start, end, 12)
		if err != nil {
			return "", err
		}
		if len(messages) > 0 {
			s.debugf("recall_hit type=yesterday count=%d", len(messages))
			sections = append(sections, "Yesterday's messages:\n"+formatMessages(messages))
		}
	}

	if shouldSearchHistory(userInput) {
		messages, err := store.SearchMessages(userInput, sessionID, userID, s.cfg.RecallLimit)
		if err != nil {
			return "", err
		}
		if len(messages) > 0 {
			s.debugf("recall_hit type=search count=%d query=%q", len(messages), userInput)
			sections = append(sections, "Relevant past messages:\n"+formatMessages(messages))
		}
	}

	return strings.TrimSpace(strings.Join(sections, "\n\n")), nil
}

func NewSessionID() string {
	return fmt.Sprintf("sess_%s", time.Now().UTC().Format("20060102_150405"))
}

func shouldSearchHistory(input string) bool {
	input = strings.TrimSpace(input)
	if input == "" {
		return false
	}

	lower := strings.ToLower(input)
	keywords := []string{"remember", "before", "last", "again", "yesterday", "week", "did we", "what did"}
	for _, keyword := range keywords {
		if strings.Contains(lower, keyword) {
			return true
		}
	}

	return len(strings.Fields(input)) >= 5
}

func yesterdayBounds() (time.Time, time.Time) {
	now := time.Now()
	y := now.AddDate(0, 0, -1)
	start := time.Date(y.Year(), y.Month(), y.Day(), 0, 0, 0, 0, y.Location())
	end := start.Add(24*time.Hour - time.Nanosecond)
	return start.UTC(), end.UTC()
}

func formatMessages(messages []storage.Message) string {
	lines := make([]string, 0, len(messages))
	for _, message := range messages {
		label := strings.ToUpper(message.Role)
		stamp := message.CreatedAt.Local().Format("2006-01-02 15:04")
		lines = append(lines, fmt.Sprintf("[%s %s] %s", stamp, label, message.Content))
	}
	return strings.Join(lines, "\n")
}

func toPromptMessages(messages []storage.Message) []llm.ChatMessage {
	out := make([]llm.ChatMessage, 0, len(messages))
	for _, message := range messages {
		out = append(out, llm.ChatMessage{
			Role:    message.Role,
			Content: message.Content,
		})
	}
	return out
}

func AgentDataPath(dataDir, tenantID, agentID string) string {
	return filepath.Join(dataDir, "tenants", tenantID, "agents", agentID)
}

func (s *Service) debugf(format string, args ...any) {
	if !s.cfg.Debug {
		return
	}
	writer := s.cfg.DebugWriter
	if writer == nil {
		writer = os.Stderr
	}
	fmt.Fprintf(writer, "[mynah-debug] "+format+"\n", args...)
}

func (s *Service) prepareEvalSandbox(tenantID, agentID, caseName string) (string, func(), error) {
	root, err := os.MkdirTemp("", "mynah-eval-*")
	if err != nil {
		return "", nil, err
	}
	targetDataDir := filepath.Join(root, "data")
	if err := os.MkdirAll(targetDataDir, 0o755); err != nil {
		_ = os.RemoveAll(root)
		return "", nil, err
	}

	source := AgentDataPath(s.cfg.DataDir, tenantID, agentID)
	target := AgentDataPath(targetDataDir, tenantID, agentID)
	if err := copyDir(source, target); err != nil {
		_ = os.RemoveAll(root)
		return "", nil, err
	}

	s.debugf("eval_sandbox case=%q path=%s", caseName, targetDataDir)
	return targetDataDir, func() { _ = os.RemoveAll(root) }, nil
}

func copyDir(source, target string) error {
	if err := os.MkdirAll(target, 0o755); err != nil {
		return err
	}

	entries, err := os.ReadDir(source)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	for _, entry := range entries {
		sourcePath := filepath.Join(source, entry.Name())
		targetPath := filepath.Join(target, entry.Name())

		info, err := entry.Info()
		if err != nil {
			return err
		}

		if entry.IsDir() {
			if err := copyDir(sourcePath, targetPath); err != nil {
				return err
			}
			continue
		}

		data, err := os.ReadFile(sourcePath)
		if err != nil {
			return err
		}
		if err := os.WriteFile(targetPath, data, info.Mode()); err != nil {
			return err
		}
	}
	return nil
}
