package secrets

import (
	"os"
	"path/filepath"
	"strings"
)

func ResolveOpenAIAPIKey() string {
	if value := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); value != "" {
		return value
	}

	raw, err := os.ReadFile(DefaultOpenAIKeyPath())
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(raw))
}

func DefaultOpenAIKeyPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ".mynah/secrets/openai_api_key"
	}
	return filepath.Join(home, ".mynah", "secrets", "openai_api_key")
}
