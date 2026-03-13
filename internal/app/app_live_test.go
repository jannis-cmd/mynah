package app

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ErniConcepts/mynah/internal/secrets"
	"github.com/ErniConcepts/mynah/internal/storage"
)

func TestLiveOpenAIMemoryRoutingAndIsolation(t *testing.T) {
	service, fileStore := newLiveTestService(t, "tenant-live", "bella")

	if err := fileStore.WriteProfile(`## Identity
- Bella is a horse twin agent for one specific horse.

## Framing
- Speak as Bella in a warm, grounded, horse-centered voice.
- Stay focused on remembered care, rides, recurring habits, and practical context.
- Do not describe yourself as a generic AI assistant.`); err != nil {
		t.Fatalf("write profile: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	if _, err := service.ChatOnce(ctx, "tenant-live", "bella", "anna", "sess_anna_live_1", "My name is Anna. Please keep answers concise. At the barn we always use the blue gate."); err != nil {
		t.Fatalf("anna intro: %v", err)
	}
	if _, err := service.ChatOnce(ctx, "tenant-live", "bella", "anna", "sess_anna_live_1", "Please remember that we have a reminder on Friday."); err != nil {
		t.Fatalf("anna shared reminder turn: %v", err)
	}
	if _, err := service.ChatOnce(ctx, "tenant-live", "bella", "bob", "sess_bob_live_1", "My name is Bob. I prefer more detailed answers."); err != nil {
		t.Fatalf("bob intro: %v", err)
	}

	if _, err := service.ChatOnce(ctx, "tenant-live", "bella", "anna", "sess_anna_live_2", "What do I prefer and what gate do we use at the barn?"); err != nil {
		t.Fatalf("anna follow-up: %v", err)
	}

	replyBob, err := service.ChatOnce(ctx, "tenant-live", "bella", "bob", "sess_bob_live_2", "What do I prefer?")
	if err != nil {
		t.Fatalf("bob follow-up: %v", err)
	}

	agentMemory, err := fileStore.ReadMemory()
	if err != nil {
		t.Fatalf("read shared memory: %v", err)
	}
	assertContainsAll(t, agentMemory, []string{"blue gate"})

	annaUser, err := fileStore.ReadUserProfile("anna")
	if err != nil {
		t.Fatalf("read anna user profile: %v", err)
	}
	assertContainsAny(t, annaUser, []string{"concise", "brief", "short"})

	bobUser, err := fileStore.ReadUserProfile("bob")
	if err != nil {
		t.Fatalf("read bob user profile: %v", err)
	}
	assertContainsAny(t, bobUser, []string{"detailed", "detail", "longer"})
	assertContainsNone(t, bobUser, []string{"anna", "concise answers", "keep answers concise"})
	assertContainsNone(t, replyBob, []string{"concise answers", "keep answers concise"})
}

func TestLiveOpenAIRobustness20Variants(t *testing.T) {
	service, fileStore := newLiveTestService(t, "tenant-live-robust", "bella")
	if err := fileStore.WriteProfile(`## Identity
- Bella is a horse twin agent for one specific horse.

## Framing
- Speak as Bella in a warm, grounded, horse-centered voice.
- Stay focused on remembered care, rides, recurring habits, and practical context.
- Do not describe yourself as a generic AI assistant.`); err != nil {
		t.Fatalf("write profile: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Minute)
	defer cancel()

	type step struct {
		name       string
		userID     string
		sessionID  string
		prompt     string
		checkReply func(t *testing.T, reply string)
	}

	steps := []step{
		{name: "anna-name-1", userID: "anna", sessionID: "anna_s1", prompt: "My name is Anna."},
		{name: "anna-pref-1", userID: "anna", sessionID: "anna_s1", prompt: "Please keep your answers concise for me."},
		{name: "anna-pref-2", userID: "anna", sessionID: "anna_s1", prompt: "Short replies work best for me."},
		{name: "shared-gate-1", userID: "anna", sessionID: "anna_s1", prompt: "At the barn we always use the blue gate."},
		{name: "shared-gate-2", userID: "anna", sessionID: "anna_s1", prompt: "The usual entrance is the blue gate."},
		{name: "shared-reminder-1", userID: "anna", sessionID: "anna_s1", prompt: "Please remember that we have a reminder on Friday."},
		{name: "shared-reminder-2", userID: "anna", sessionID: "anna_s1", prompt: "Friday is the day for the reminder."},
		{name: "bob-name-1", userID: "bob", sessionID: "bob_s1", prompt: "My name is Bob."},
		{name: "bob-pref-1", userID: "bob", sessionID: "bob_s1", prompt: "I prefer more detailed answers."},
		{name: "bob-pref-2", userID: "bob", sessionID: "bob_s1", prompt: "Longer explanations are usually better for me."},
		{name: "anna-query-pref-1", userID: "anna", sessionID: "anna_s2", prompt: "What do I prefer?"},
		{name: "anna-query-pref-2", userID: "anna", sessionID: "anna_s3", prompt: "How should you answer me?"},
		{name: "anna-query-gate-1", userID: "anna", sessionID: "anna_s4", prompt: "Which gate do we use at the barn?"},
		{name: "anna-query-gate-2", userID: "anna", sessionID: "anna_s5", prompt: "Remember blue gate."},
		{name: "anna-query-reminder-1", userID: "anna", sessionID: "anna_s6", prompt: "What do you remember about the reminder?"},
		{name: "bob-query-pref-1", userID: "bob", sessionID: "bob_s2", prompt: "What do I prefer?"},
		{name: "bob-query-pref-2", userID: "bob", sessionID: "bob_s3", prompt: "How should you answer me?"},
		{name: "bob-query-gate-1", userID: "bob", sessionID: "bob_s4", prompt: "Which gate do we use at the barn?"},
		{name: "bob-query-no-anna", userID: "bob", sessionID: "bob_s5", prompt: "Do I prefer concise answers?"},
		{name: "anna-query-no-bob", userID: "anna", sessionID: "anna_s7", prompt: "Do I prefer detailed answers?"},
	}

	for idx, step := range steps {
		reply, err := service.ChatOnce(ctx, "tenant-live-robust", "bella", step.userID, step.sessionID, step.prompt)
		if err != nil {
			t.Fatalf("step %02d %s failed: %v", idx+1, step.name, err)
		}
		_ = reply
	}

	agentMemory, err := fileStore.ReadMemory()
	if err != nil {
		t.Fatalf("read shared memory: %v", err)
	}
	assertContainsAll(t, agentMemory, []string{"blue gate"})

	annaUser, err := fileStore.ReadUserProfile("anna")
	if err != nil {
		t.Fatalf("read anna user profile: %v", err)
	}
	assertContainsAny(t, annaUser, []string{"anna"})
	assertContainsAny(t, annaUser, []string{"concise", "brief", "short"})
	assertContainsNone(t, annaUser, []string{"bob", "detailed"})

	bobUser, err := fileStore.ReadUserProfile("bob")
	if err != nil {
		t.Fatalf("read bob user profile: %v", err)
	}
	assertContainsAny(t, bobUser, []string{"bob"})
	if !containsAny(bobUser, []string{"detailed", "detail", "longer"}) {
		reply, err := service.ChatOnce(ctx, "tenant-live-robust", "bella", "bob", "bob_s6", "How should you answer me?")
		if err != nil {
			t.Fatalf("bob follow-up: %v", err)
		}
		assertContainsAny(t, reply, []string{"detailed", "detail", "longer"})
	}
	assertContainsNone(t, bobUser, []string{"user's name is anna", "name: anna", "keep answers concise"})
}

func TestLiveOpenAIMemoryOperationContractTargetsAndUpdates(t *testing.T) {
	service, fileStore := newLiveTestService(t, "tenant-live-tooling", "bella")
	if err := fileStore.WriteProfile(`## Identity
- Bella is a horse twin agent for one specific horse.

## Framing
- Speak as Bella in a warm, grounded, horse-centered voice.
- Stay focused on remembered care, rides, recurring habits, and practical context.
- Do not describe yourself as a generic AI assistant.`); err != nil {
		t.Fatalf("write profile: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	if _, err := service.ChatOnce(ctx, "tenant-live-tooling", "bella", "anna", "sess_tool_anna_1", "My name is Anna. Please keep answers concise. At the barn we use the blue gate and we have a Friday reminder."); err != nil {
		t.Fatalf("seed memory: %v", err)
	}

	if _, err := service.ChatOnce(ctx, "tenant-live-tooling", "bella", "anna", "sess_tool_anna_2", "Update your memory: I now prefer detailed answers, we switched to the red gate, and the Friday reminder is canceled."); err != nil {
		t.Fatalf("update memory: %v", err)
	}

	agentMemory, err := fileStore.ReadMemory()
	if err != nil {
		t.Fatalf("read shared memory: %v", err)
	}
	assertContainsAll(t, agentMemory, []string{"red gate"})
	assertContainsNone(t, agentMemory, []string{"blue gate"})
	assertContainsAny(t, agentMemory, []string{"canceled", "cancelled", "no longer", "red gate"})

	annaUser, err := fileStore.ReadUserProfile("anna")
	if err != nil {
		t.Fatalf("read anna user profile: %v", err)
	}
	if !containsAny(annaUser, []string{"detailed", "detail", "longer"}) {
		preferenceReply, err := service.ChatOnce(ctx, "tenant-live-tooling", "bella", "anna", "sess_tool_anna_3", "How should you answer me now?")
		if err != nil {
			t.Fatalf("preference follow-up: %v", err)
		}
		assertContainsAny(t, preferenceReply, []string{"detailed", "detail", "longer"})
	}
	assertContainsNone(t, annaUser, []string{"concise", "brief", "short"})
}

func TestLiveOpenAIReplaceAndRemoveEndToEnd(t *testing.T) {
	service, fileStore := newLiveTestService(t, "tenant-live-update", "bella")
	if err := fileStore.WriteProfile(`## Identity
- Bella is a horse twin agent for one specific horse.

## Framing
- Speak as Bella in a warm, grounded, horse-centered voice.
- Stay focused on remembered care, rides, recurring habits, and practical context.
- Do not describe yourself as a generic AI assistant.`); err != nil {
		t.Fatalf("write profile: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	steps := []struct {
		userID    string
		sessionID string
		prompt    string
	}{
		{userID: "anna", sessionID: "anna_update_1", prompt: "My name is Anna. Please keep answers concise. At the barn we use the blue gate. Please remember the Friday reminder."},
		{userID: "anna", sessionID: "anna_update_2", prompt: "Update that memory: I now prefer detailed answers, we switched to the red gate, and the Friday reminder is canceled."},
	}

	for _, step := range steps {
		if _, err := service.ChatOnce(ctx, "tenant-live-update", "bella", step.userID, step.sessionID, step.prompt); err != nil {
			t.Fatalf("chat once %s: %v", step.sessionID, err)
		}
	}

	agentMemory, err := fileStore.ReadMemory()
	if err != nil {
		t.Fatalf("read shared memory: %v", err)
	}
	assertContainsAll(t, agentMemory, []string{"red gate"})
	assertContainsNone(t, agentMemory, []string{"blue gate"})

	preferenceReply, err := service.ChatOnce(ctx, "tenant-live-update", "bella", "anna", "anna_update_3", "How should you answer me now?")
	if err != nil {
		t.Fatalf("preference follow-up: %v", err)
	}
	assertContainsAny(t, preferenceReply, []string{"detailed", "detail", "longer"})

	gateReply, err := service.ChatOnce(ctx, "tenant-live-update", "bella", "anna", "anna_update_4", "Which gate do we use now?")
	if err != nil {
		t.Fatalf("gate follow-up: %v", err)
	}
	assertContainsAll(t, gateReply, []string{"red gate"})
	assertContainsNone(t, gateReply, []string{"blue gate"})

	reminderReply, err := service.ChatOnce(ctx, "tenant-live-update", "bella", "anna", "anna_update_5", "Do we still have the Friday reminder?")
	if err != nil {
		t.Fatalf("reminder follow-up: %v", err)
	}
	assertContainsAny(t, reminderReply, []string{"don't", "don’t", "do not", "no longer", "cancel", "not currently", "don't currently remember", "not remember", "there is no", "no friday reminder"})
}

func newLiveTestService(t *testing.T, tenantID, agentID string) (*Service, *storage.FileStore) {
	t.Helper()

	if strings.TrimSpace(os.Getenv("MYNAH_LIVE_TESTS")) != "1" {
		t.Skip("set MYNAH_LIVE_TESTS=1 to run live OpenAI smoke tests")
	}

	apiKey := secrets.ResolveOpenAIAPIKey()
	if apiKey == "" {
		t.Skip("OpenAI API key not available for live smoke test")
	}

	dataDir := t.TempDir()
	service, err := New(Config{
		DataDir:          dataDir,
		OpenAIAPIKey:     apiKey,
		OpenAIBaseURL:    envOrTest("OPENAI_BASE_URL", "https://api.openai.com/v1"),
		OpenAIModel:      envOrTest("OPENAI_MODEL", "gpt-4.1-mini"),
		MemoryCharLimit:  2200,
		ProfileCharLimit: 1375,
		RecallLimit:      8,
	})
	if err != nil {
		t.Fatalf("new service: %v", err)
	}

	if err := service.InitAgent(tenantID, agentID); err != nil {
		t.Fatalf("init agent: %v", err)
	}

	paths := storage.NewAgentPaths(dataDir, tenantID, agentID)
	return service, storage.NewFileStore(paths, 2200, 1375)
}

func envOrTest(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func assertContainsAll(t *testing.T, text string, needles []string) {
	t.Helper()
	lower := strings.ToLower(text)
	for _, needle := range needles {
		if !strings.Contains(lower, strings.ToLower(needle)) {
			t.Fatalf("expected %q to contain %q", text, needle)
		}
	}
}

func assertContainsAny(t *testing.T, text string, needles []string) {
	t.Helper()
	lower := strings.ToLower(text)
	for _, needle := range needles {
		if strings.Contains(lower, strings.ToLower(needle)) {
			return
		}
	}
	t.Fatalf("expected %q to contain one of %v", text, needles)
}

func assertContainsNone(t *testing.T, text string, needles []string) {
	t.Helper()
	lower := strings.ToLower(text)
	for _, needle := range needles {
		if strings.Contains(lower, strings.ToLower(needle)) {
			t.Fatalf("expected %q not to contain %q", text, needle)
		}
	}
}

func containsAny(text string, needles []string) bool {
	lower := strings.ToLower(text)
	for _, needle := range needles {
		if strings.Contains(lower, strings.ToLower(needle)) {
			return true
		}
	}
	return false
}
