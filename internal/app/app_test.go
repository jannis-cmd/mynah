package app

import (
	"context"
	"strings"
	"testing"

	"github.com/ErniConcepts/mynah/internal/llm"
	"github.com/ErniConcepts/mynah/internal/storage"
)

type fakeLLMClient struct{}

func (fakeLLMClient) GenerateReply(_ context.Context, request llm.ReplyRequest) (string, error) {
	lastUser := lastUserMessage(request.RecentMessages)
	lower := strings.ToLower(lastUser)
	prefix := ""
	if strings.Contains(strings.ToLower(request.ProfileDoc), "bella") {
		prefix = "Bella: "
	}

	parts := make([]string, 0, 3)
	if strings.Contains(lower, "what do i prefer") {
		if strings.Contains(strings.ToLower(request.UserDoc), "concise answers") {
			parts = append(parts, "You prefer concise answers.")
		} else {
			parts = append(parts, "I do not know your preferences yet.")
		}
	}
	if strings.Contains(lower, "what did we do yesterday") {
		if strings.TrimSpace(request.RecallContext) != "" {
			parts = append(parts, "Yesterday we discussed the blue gate.")
		} else {
			parts = append(parts, "I do not have recall for that user yet.")
		}
	}
	if strings.Contains(lower, "remember blue gate") {
		if strings.TrimSpace(request.RecallContext) != "" {
			parts = append(parts, "You told me about the blue gate.")
		} else {
			parts = append(parts, "I do not have recall for that user yet.")
		}
	}
	if strings.Contains(lower, "what did you decide") {
		if strings.Contains(strings.ToLower(request.MemoryDoc), "<decision> reminder promised for friday.") {
			parts = append(parts, "I decided to promise a reminder for Friday.")
		} else {
			parts = append(parts, "I have no stored decision for that yet.")
		}
	}
	if strings.Contains(lower, "what do we use at the barn") {
		if strings.Contains(strings.ToLower(request.MemoryDoc), "blue gate") {
			parts = append(parts, "We use the blue gate at the barn.")
		} else {
			parts = append(parts, "I do not have that shared memory yet.")
		}
	}
	if len(parts) == 0 {
		parts = append(parts, "Noted.")
	}
	return prefix + strings.Join(parts, " "), nil
}

func (fakeLLMClient) ReviseMemory(_ context.Context, request llm.MemoryRevisionRequest) (llm.MemoryRevision, error) {
	memoryDoc := request.MemoryDoc
	userDoc := request.UserDoc
	lower := strings.ToLower(request.UserMessage)

	if strings.Contains(lower, "my name is anna") {
		userDoc = addMemoryLine(userDoc, "<memory> Name: Anna.")
	}
	if strings.Contains(lower, "i like concise answers") {
		userDoc = addMemoryLine(userDoc, "<memory> Prefers concise answers.")
	}
	if strings.Contains(lower, "my name is bob") {
		userDoc = addMemoryLine(userDoc, "<memory> Name: Bob.")
	}
	if strings.Contains(lower, "i prefer detailed answers") {
		userDoc = addMemoryLine(userDoc, "<memory> Prefers detailed answers.")
	}
	if strings.Contains(lower, "blue gate") {
		memoryDoc = addMemoryLine(memoryDoc, "<memory> The barn uses the blue gate.")
	}
	if strings.Contains(lower, "reminder on friday") {
		memoryDoc = addMemoryLine(memoryDoc, "<decision> Reminder promised for Friday.")
	}

	return llm.MemoryRevision{
		MemoryDoc: memoryDoc,
		UserDoc:   userDoc,
		Reason:    "test revision",
	}, nil
}

func TestChatOnceRequiresUserID(t *testing.T) {
	service := &Service{
		cfg: Config{DataDir: t.TempDir(), MemoryCharLimit: 2200, ProfileCharLimit: 1375, RecallLimit: 8},
		llmClient: fakeLLMClient{},
	}

	_, err := service.ChatOnce(context.Background(), "tenant", "agent", "", "sess_1", "hello")
	if err == nil || !strings.Contains(err.Error(), "user_id is required") {
		t.Fatalf("expected missing user_id error, got %v", err)
	}
}

func TestUserScopedMemoryRoutingAndIsolation(t *testing.T) {
	service, paths := newTestService(t)
	if err := service.InitAgent("tenant", "bella"); err != nil {
		t.Fatalf("init agent: %v", err)
	}

	fileStore := storage.NewFileStore(paths, 2200, 1375)
	if err := fileStore.WriteProfile("## Identity\n- Bella is a horse twin agent.\n- Speak as Bella in a warm, grounded voice."); err != nil {
		t.Fatalf("write profile: %v", err)
	}

	sessionAnna := "sess_anna_1"
	if _, err := service.ChatOnce(context.Background(), "tenant", "bella", "anna", sessionAnna, "My name is Anna and I like concise answers."); err != nil {
		t.Fatalf("anna intro: %v", err)
	}
	if _, err := service.ChatOnce(context.Background(), "tenant", "bella", "anna", sessionAnna, "The barn uses the blue gate."); err != nil {
		t.Fatalf("shared memory write: %v", err)
	}
	if _, err := service.ChatOnce(context.Background(), "tenant", "bella", "anna", sessionAnna, "Please remember that you promised me a reminder on Friday."); err != nil {
		t.Fatalf("decision write: %v", err)
	}

	sessionBob := "sess_bob_1"
	if _, err := service.ChatOnce(context.Background(), "tenant", "bella", "bob", sessionBob, "My name is Bob and I prefer detailed answers."); err != nil {
		t.Fatalf("bob intro: %v", err)
	}

	agentMemory, err := fileStore.ReadMemory()
	if err != nil {
		t.Fatalf("read memory: %v", err)
	}
	if !strings.Contains(agentMemory, "<memory> The barn uses the blue gate.") {
		t.Fatalf("expected shared memory entry, got %q", agentMemory)
	}
	if !strings.Contains(agentMemory, "<decision> Reminder promised for Friday.") {
		t.Fatalf("expected decision entry, got %q", agentMemory)
	}

	annaUser, err := fileStore.ReadUserProfile("anna")
	if err != nil {
		t.Fatalf("read anna user doc: %v", err)
	}
	if !strings.Contains(annaUser, "Name: Anna.") || !strings.Contains(annaUser, "Prefers concise answers.") {
		t.Fatalf("expected anna user doc to contain scoped memory, got %q", annaUser)
	}

	bobUser, err := fileStore.ReadUserProfile("bob")
	if err != nil {
		t.Fatalf("read bob user doc: %v", err)
	}
	if !strings.Contains(bobUser, "Name: Bob.") || !strings.Contains(bobUser, "Prefers detailed answers.") {
		t.Fatalf("expected bob user doc to contain scoped memory, got %q", bobUser)
	}
	if strings.Contains(bobUser, "Anna") {
		t.Fatalf("bob user doc leaked anna data: %q", bobUser)
	}
}

func TestRecallAndReplyStayUserScopedAndCoherent(t *testing.T) {
	service, paths := newTestService(t)
	if err := service.InitAgent("tenant", "bella"); err != nil {
		t.Fatalf("init agent: %v", err)
	}

	fileStore := storage.NewFileStore(paths, 2200, 1375)
	if err := fileStore.WriteProfile("## Identity\n- Bella is a horse twin agent.\n- Speak as Bella in a warm, grounded voice."); err != nil {
		t.Fatalf("write profile: %v", err)
	}

	if _, err := service.ChatOnce(context.Background(), "tenant", "bella", "anna", "sess_anna_1", "My name is Anna and I like concise answers."); err != nil {
		t.Fatalf("anna intro: %v", err)
	}
	if _, err := service.ChatOnce(context.Background(), "tenant", "bella", "anna", "sess_anna_1", "The barn uses the blue gate."); err != nil {
		t.Fatalf("anna shared memory: %v", err)
	}

	replyAnna, err := service.ChatOnce(context.Background(), "tenant", "bella", "anna", "sess_anna_2", "What do I prefer?")
	if err != nil {
		t.Fatalf("anna follow-up: %v", err)
	}
	if !strings.Contains(replyAnna, "Bella:") {
		t.Fatalf("expected reply aligned with profile framing, got %q", replyAnna)
	}
	if !strings.Contains(replyAnna, "You prefer concise answers.") {
		t.Fatalf("expected anna user memory in reply, got %q", replyAnna)
	}

	recallAnna, err := service.ChatOnce(context.Background(), "tenant", "bella", "anna", "sess_anna_3", "Remember blue gate.")
	if err != nil {
		t.Fatalf("anna recall follow-up: %v", err)
	}
	if !strings.Contains(recallAnna, "You told me about the blue gate.") {
		t.Fatalf("expected anna recall in reply, got %q", recallAnna)
	}

	replyBob, err := service.ChatOnce(context.Background(), "tenant", "bella", "bob", "sess_bob_1", "Remember blue gate.")
	if err != nil {
		t.Fatalf("bob follow-up: %v", err)
	}
	if strings.Contains(replyBob, "blue gate") {
		t.Fatalf("expected no cross-user recall leakage, got %q", replyBob)
	}
}

func newTestService(t *testing.T) (*Service, storage.AgentPaths) {
	t.Helper()

	dataDir := t.TempDir()
	service := &Service{
		cfg: Config{
			DataDir:          dataDir,
			MemoryCharLimit:  2200,
			ProfileCharLimit: 1375,
			RecallLimit:      8,
		},
		llmClient: fakeLLMClient{},
	}
	return service, storage.NewAgentPaths(dataDir, "tenant", "bella")
}

func addMemoryLine(doc, line string) string {
	line = strings.TrimSpace(line)
	if line == "" || strings.Contains(doc, line) {
		return doc
	}
	if strings.TrimSpace(doc) == "" {
		return line
	}
	return strings.TrimSpace(doc) + "\n" + line
}

func lastUserMessage(messages []llm.ChatMessage) string {
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role == "user" {
			return messages[index].Content
		}
	}
	return ""
}
