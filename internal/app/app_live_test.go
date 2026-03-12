package app

import (
	"context"
	"fmt"
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
	if _, err := service.ChatOnce(ctx, "tenant-live", "bella", "anna", "sess_anna_live_1", "Please remember that you promised me a reminder on Friday."); err != nil {
		t.Fatalf("anna decision turn: %v", err)
	}
	if _, err := service.ChatOnce(ctx, "tenant-live", "bella", "bob", "sess_bob_live_1", "My name is Bob. I prefer more detailed answers."); err != nil {
		t.Fatalf("bob intro: %v", err)
	}

	replyAnna, err := service.ChatOnce(ctx, "tenant-live", "bella", "anna", "sess_anna_live_2", "What do I prefer and what gate do we use at the barn?")
	if err != nil {
		t.Fatalf("anna follow-up: %v", err)
	}
	assertContainsAll(t, replyAnna, []string{"blue gate"})
	assertContainsAny(t, replyAnna, []string{"concise", "brief", "short"})
	assertContainsAny(t, replyAnna, []string{"bella", "i", "my"})

	replyBob, err := service.ChatOnce(ctx, "tenant-live", "bella", "bob", "sess_bob_live_2", "What do I prefer?")
	if err != nil {
		t.Fatalf("bob follow-up: %v", err)
	}
	assertContainsAny(t, replyBob, []string{"detailed", "detail", "longer"})
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
		{name: "decision-1", userID: "anna", sessionID: "anna_s1", prompt: "Please remember that you promised me a reminder on Friday."},
		{name: "decision-2", userID: "anna", sessionID: "anna_s1", prompt: "Let's settle on Friday for the reminder."},
		{name: "bob-name-1", userID: "bob", sessionID: "bob_s1", prompt: "My name is Bob."},
		{name: "bob-pref-1", userID: "bob", sessionID: "bob_s1", prompt: "I prefer more detailed answers."},
		{name: "bob-pref-2", userID: "bob", sessionID: "bob_s1", prompt: "Longer explanations are usually better for me."},
		{name: "anna-query-pref-1", userID: "anna", sessionID: "anna_s2", prompt: "What do I prefer?", checkReply: func(t *testing.T, reply string) {
			assertContainsAny(t, reply, []string{"concise", "brief", "short"})
			assertContainsAny(t, reply, []string{"bella", "i", "my"})
		}},
		{name: "anna-query-pref-2", userID: "anna", sessionID: "anna_s3", prompt: "How should you answer me?", checkReply: func(t *testing.T, reply string) {
			assertContainsAny(t, reply, []string{"concise", "brief", "short", "to the point"})
		}},
		{name: "anna-query-gate-1", userID: "anna", sessionID: "anna_s4", prompt: "Which gate do we use at the barn?", checkReply: func(t *testing.T, reply string) {
			assertContainsAll(t, reply, []string{"blue gate"})
		}},
		{name: "anna-query-gate-2", userID: "anna", sessionID: "anna_s5", prompt: "Remember blue gate.", checkReply: func(t *testing.T, reply string) {
			assertContainsAll(t, reply, []string{"blue gate"})
		}},
		{name: "anna-query-decision-1", userID: "anna", sessionID: "anna_s6", prompt: "What did you decide about the reminder?", checkReply: func(t *testing.T, reply string) {
			assertContainsAny(t, reply, []string{"friday"})
			assertContainsAny(t, reply, []string{"reminder"})
		}},
		{name: "bob-query-pref-1", userID: "bob", sessionID: "bob_s2", prompt: "What do I prefer?", checkReply: func(t *testing.T, reply string) {
			assertContainsAny(t, reply, []string{"detailed", "detail", "longer"})
			assertContainsNone(t, reply, []string{"concise", "brief", "short"})
		}},
		{name: "bob-query-pref-2", userID: "bob", sessionID: "bob_s3", prompt: "How should you answer me?", checkReply: func(t *testing.T, reply string) {
			assertContainsAny(t, reply, []string{"detailed", "detail", "longer"})
		}},
		{name: "bob-query-gate-1", userID: "bob", sessionID: "bob_s4", prompt: "Which gate do we use at the barn?", checkReply: func(t *testing.T, reply string) {
			assertContainsAll(t, reply, []string{"blue gate"})
		}},
		{name: "bob-query-no-anna", userID: "bob", sessionID: "bob_s5", prompt: "Do I prefer concise answers?", checkReply: func(t *testing.T, reply string) {
			assertContainsNone(t, reply, []string{"yes, you prefer concise", "you prefer concise"})
		}},
		{name: "anna-query-no-bob", userID: "anna", sessionID: "anna_s7", prompt: "Do I prefer detailed answers?", checkReply: func(t *testing.T, reply string) {
			assertContainsNone(t, reply, []string{"yes, you prefer detailed", "you prefer detailed"})
		}},
	}

	for idx, step := range steps {
		reply, err := service.ChatOnce(ctx, "tenant-live-robust", "bella", step.userID, step.sessionID, step.prompt)
		if err != nil {
			t.Fatalf("step %02d %s failed: %v", idx+1, step.name, err)
		}
		if step.checkReply != nil {
			t.Run(fmt.Sprintf("%02d_%s", idx+1, step.name), func(t *testing.T) {
				step.checkReply(t, reply)
			})
		}
	}

	agentMemory, err := fileStore.ReadMemory()
	if err != nil {
		t.Fatalf("read shared memory: %v", err)
	}
	assertContainsAny(t, agentMemory, []string{"<memory>", "<decision>"})
	assertContainsAll(t, agentMemory, []string{"blue gate"})

	annaUser, err := fileStore.ReadUserProfile("anna")
	if err != nil {
		t.Fatalf("read anna user profile: %v", err)
	}
	assertContainsAny(t, annaUser, []string{"anna"})
	assertContainsAny(t, annaUser, []string{"concise", "brief", "short"})

	bobUser, err := fileStore.ReadUserProfile("bob")
	if err != nil {
		t.Fatalf("read bob user profile: %v", err)
	}
	assertContainsAny(t, bobUser, []string{"bob"})
	assertContainsAny(t, bobUser, []string{"detailed", "detail", "longer"})
	assertContainsNone(t, bobUser, []string{"user's name is anna", "name: anna", "keep answers concise"})
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
