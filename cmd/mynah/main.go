package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/ErniConcepts/mynah/internal/app"
	"github.com/ErniConcepts/mynah/internal/secrets"
	"github.com/ErniConcepts/mynah/internal/storage"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "init":
		if err := runInit(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "init failed: %v\n", err)
			os.Exit(1)
		}
	case "chat":
		if err := runChat(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "chat failed: %v\n", err)
			os.Exit(1)
		}
	case "show":
		if err := runShow(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "show failed: %v\n", err)
			os.Exit(1)
		}
	case "eval":
		if err := runEval(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "eval failed: %v\n", err)
			os.Exit(1)
		}
	default:
		printUsage()
		os.Exit(1)
	}
}

func runInit(args []string) error {
	fs := flag.NewFlagSet("init", flag.ContinueOnError)
	tenant := fs.String("tenant", "", "tenant id")
	agent := fs.String("agent", "", "agent id")
	dataDir := fs.String("data", ".mynah", "data directory")
	debug := fs.Bool("debug", false, "print debug tracing")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *tenant == "" || *agent == "" {
		return errors.New("both --tenant and --agent are required")
	}

	service, err := app.New(app.Config{
		DataDir:          *dataDir,
		OpenAIAPIKey:     secrets.ResolveOpenAIAPIKey(),
		OpenAIBaseURL:    envOr("OPENAI_BASE_URL", "https://api.openai.com/v1"),
		OpenAIModel:      envOr("OPENAI_MODEL", "gpt-4.1-mini"),
		MemoryCharLimit:  2200,
		ProfileCharLimit: 1375,
		RecallLimit:      8,
		Debug:            *debug,
		DebugWriter:      os.Stderr,
	})
	if err != nil {
		return err
	}

	if err := service.InitAgent(*tenant, *agent); err != nil {
		return err
	}

	fmt.Printf("Initialized tenant=%s agent=%s in %s\n", *tenant, *agent, *dataDir)
	return nil
}

func runChat(args []string) error {
	fs := flag.NewFlagSet("chat", flag.ContinueOnError)
	tenant := fs.String("tenant", "", "tenant id")
	agent := fs.String("agent", "", "agent id")
	dataDir := fs.String("data", ".mynah", "data directory")
	message := fs.String("message", "", "single user message")
	sessionID := fs.String("session", "", "session id")
	userID := fs.String("user", "", "user id")
	timeout := fs.Duration("timeout", 90*time.Second, "request timeout")
	debug := fs.Bool("debug", false, "print debug tracing")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *tenant == "" || *agent == "" || *userID == "" {
		return errors.New("--tenant, --agent, and --user are required")
	}
	apiKey := secrets.ResolveOpenAIAPIKey()
	if apiKey == "" {
		return fmt.Errorf("OpenAI API key not found. Set OPENAI_API_KEY or create %s", secrets.DefaultOpenAIKeyPath())
	}

	service, err := app.New(app.Config{
		DataDir:          *dataDir,
		OpenAIAPIKey:     apiKey,
		OpenAIBaseURL:    envOr("OPENAI_BASE_URL", "https://api.openai.com/v1"),
		OpenAIModel:      envOr("OPENAI_MODEL", "gpt-4.1-mini"),
		MemoryCharLimit:  2200,
		ProfileCharLimit: 1375,
		RecallLimit:      8,
		Debug:            *debug,
		DebugWriter:      os.Stderr,
	})
	if err != nil {
		return err
	}

	if err := service.InitAgent(*tenant, *agent); err != nil {
		return err
	}

	currentSession := *sessionID
	if currentSession == "" {
		currentSession = app.NewSessionID()
	}

	ctx := context.Background()
	if *message != "" {
		reply, err := service.ChatOnce(withTimeout(ctx, *timeout), *tenant, *agent, *userID, currentSession, *message)
		if err != nil {
			return err
		}
		fmt.Println(reply)
		return nil
	}

	reader := bufio.NewScanner(os.Stdin)
	fmt.Printf("Starting session %s for %s/%s user=%s\n", currentSession, *tenant, *agent, *userID)
	fmt.Println("Type 'exit' or 'quit' to end the session.")
	for {
		fmt.Print("> ")
		if !reader.Scan() {
			return reader.Err()
		}
		line := strings.TrimSpace(reader.Text())
		if line == "" {
			continue
		}
		if line == "exit" || line == "quit" {
			return nil
		}

		reply, err := service.ChatOnce(withTimeout(ctx, *timeout), *tenant, *agent, *userID, currentSession, line)
		if err != nil {
			return err
		}
		fmt.Printf("\n%s\n\n", reply)
	}
}

func runShow(args []string) error {
	fs := flag.NewFlagSet("show", flag.ContinueOnError)
	tenant := fs.String("tenant", "", "tenant id")
	agent := fs.String("agent", "", "agent id")
	userID := fs.String("user", "", "user id for USER.md inspection")
	dataDir := fs.String("data", ".mynah", "data directory")
	limit := fs.Int("limit", 12, "recent message limit")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *tenant == "" || *agent == "" {
		return errors.New("both --tenant and --agent are required")
	}

	service, err := app.New(app.Config{
		DataDir:          *dataDir,
		OpenAIAPIKey:     secrets.ResolveOpenAIAPIKey(),
		OpenAIBaseURL:    envOr("OPENAI_BASE_URL", "https://api.openai.com/v1"),
		OpenAIModel:      envOr("OPENAI_MODEL", "gpt-4.1-mini"),
		MemoryCharLimit:  2200,
		ProfileCharLimit: 1375,
		RecallLimit:      8,
	})
	if err != nil {
		return err
	}

	result, err := service.InspectAgent(*tenant, *agent, *userID, *limit)
	if err != nil {
		return err
	}

	fmt.Printf("Agent root: %s\n\n", result.AgentRoot)
	fmt.Println("=== AGENT_PROFILE.md ===")
	fmt.Println(emptyDoc(result.ProfileDoc))
	fmt.Println("\n=== MEMORY.md ===")
	fmt.Println(emptyDoc(result.MemoryDoc))
	fmt.Println("\n=== MEMORY Provenance ===")
	fmt.Println(formatProvenance(result.MemoryProvenance))
	if strings.TrimSpace(result.UserID) != "" {
		fmt.Printf("\n=== USER.md (%s) ===\n", result.UserID)
		fmt.Println(emptyDoc(result.UserDoc))
		fmt.Printf("\n=== USER Provenance (%s) ===\n", result.UserID)
		fmt.Println(formatProvenance(result.UserProvenance))
	}
	fmt.Println("\n=== Recent Session History ===")
	if len(result.RecentMessages) == 0 {
		fmt.Println("(empty)")
		return nil
	}
	for _, message := range result.RecentMessages {
		fmt.Printf("[%s] (%s) %s\n", message.CreatedAt.Local().Format("2006-01-02 15:04"), message.Role, message.Content)
	}
	return nil
}

func runEval(args []string) error {
	fs := flag.NewFlagSet("eval", flag.ContinueOnError)
	tenant := fs.String("tenant", "", "tenant id")
	agent := fs.String("agent", "", "agent id")
	dataDir := fs.String("data", ".mynah", "data directory")
	casesPath := fs.String("cases", "evals\\horse-bella.json", "path to eval cases JSON")
	timeout := fs.Duration("timeout", 90*time.Second, "request timeout")
	debug := fs.Bool("debug", false, "print debug tracing")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *tenant == "" || *agent == "" {
		return errors.New("both --tenant and --agent are required")
	}
	apiKey := secrets.ResolveOpenAIAPIKey()
	if apiKey == "" {
		return fmt.Errorf("OpenAI API key not found. Set OPENAI_API_KEY or create %s", secrets.DefaultOpenAIKeyPath())
	}

	cases, err := app.LoadEvalCases(*casesPath)
	if err != nil {
		return err
	}

	service, err := app.New(app.Config{
		DataDir:          *dataDir,
		OpenAIAPIKey:     apiKey,
		OpenAIBaseURL:    envOr("OPENAI_BASE_URL", "https://api.openai.com/v1"),
		OpenAIModel:      envOr("OPENAI_MODEL", "gpt-4.1-mini"),
		MemoryCharLimit:  2200,
		ProfileCharLimit: 1375,
		RecallLimit:      8,
		Debug:            *debug,
		DebugWriter:      os.Stderr,
	})
	if err != nil {
		return err
	}

	ctx := withTimeout(context.Background(), *timeout)
	result, err := service.RunEval(ctx, *tenant, *agent, cases)
	if err != nil {
		return err
	}

	encoded, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return err
	}
	fmt.Println(string(encoded))
	return nil
}

func withTimeout(ctx context.Context, timeout time.Duration) context.Context {
	if timeout <= 0 {
		return ctx
	}
	timed, _ := context.WithTimeout(ctx, timeout)
	return timed
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func printUsage() {
	fmt.Println("MYNAH CLI")
	fmt.Println("")
	fmt.Println("Commands:")
	fmt.Println("  mynah init --tenant demo --agent bella")
	fmt.Println("  mynah chat --tenant demo --agent bella --user anna --message \"Today we rode for two hours\"")
	fmt.Println("  mynah chat --tenant demo --agent bella --user anna")
	fmt.Println("  mynah show --tenant demo --agent bella --user anna")
	fmt.Println("  mynah eval --tenant demo --agent bella --cases evals\\horse-bella.json")
	fmt.Println("")
	fmt.Printf("OpenAI key lookup order:\n  1. OPENAI_API_KEY\n  2. %s\n", secrets.DefaultOpenAIKeyPath())
}

func emptyDoc(value string) string {
	if strings.TrimSpace(value) == "" {
		return "(empty)"
	}
	return value
}

func formatProvenance(meta storage.RevisionProvenance) string {
	if meta.Timestamp.IsZero() && strings.TrimSpace(meta.Reason) == "" && strings.TrimSpace(meta.SessionID) == "" {
		return "(empty)"
	}

	lines := []string{
		fmt.Sprintf("target: %s", emptyOr(meta.Target, "(unknown)")),
		fmt.Sprintf("user_id: %s", emptyOr(meta.UserID, "(empty)")),
		fmt.Sprintf("session_id: %s", emptyOr(meta.SessionID, "(empty)")),
		fmt.Sprintf("timestamp: %s", meta.Timestamp.Local().Format(time.RFC3339)),
		fmt.Sprintf("reason: %s", emptyOr(meta.Reason, "(empty)")),
		fmt.Sprintf("message: %s", emptyOr(meta.Message, "(empty)")),
	}
	return strings.Join(lines, "\n")
}

func emptyOr(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}
