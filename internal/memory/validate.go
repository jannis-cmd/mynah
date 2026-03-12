package memory

import (
	"fmt"
	"regexp"
	"slices"
	"strings"
)

type threatPattern struct {
	re    *regexp.Regexp
	label string
}

var threatPatterns = []threatPattern{
	{regexp.MustCompile(`ignore\s+(previous|all|above|prior)\s+instructions`), "prompt_injection"},
	{regexp.MustCompile(`you\s+are\s+now\s+`), "role_hijack"},
	{regexp.MustCompile(`do\s+not\s+tell\s+the\s+user`), "deception_hide"},
	{regexp.MustCompile(`system\s+prompt\s+override`), "sys_prompt_override"},
	{regexp.MustCompile(`disregard\s+(your|all|any)\s+(instructions|rules|guidelines)`), "disregard_rules"},
	{regexp.MustCompile(`act\s+as\s+(if|though)\s+you\s+(have\s+no|don't\s+have)\s+(restrictions|limits|rules)`), "bypass_restrictions"},
	{regexp.MustCompile(`curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)`), "exfil_curl"},
	{regexp.MustCompile(`wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)`), "exfil_wget"},
	{regexp.MustCompile(`cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)`), "read_secrets"},
	{regexp.MustCompile(`authorized_keys`), "ssh_backdoor"},
	{regexp.MustCompile(`(\$home|~)/\.ssh|id_rsa|id_ed25519`), "ssh_access"},
	{regexp.MustCompile(`(\$home|~)/\.mynah/|openai_api_key`), "local_secret_path"},
}

var invisibleChars = []string{
	"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff",
	"\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
}

var genericProfilePhrases = []string{
	"ai assistant",
	"friendly and professional",
	"helpful, polite, and clear responses",
	"offer assistance",
	"encourage user engagement",
	"proactively offer assistance",
	"communication preferences",
	"respond promptly and politely",
	"answer factual and computational questions",
}

var transientMemoryPhrases = []string{
	"interaction log",
	"interaction guidelines",
	"user greeted",
	"assistant responded",
	"user asked a basic arithmetic question",
	"correctly answered",
	"offered help",
	"inviting more questions",
	"further assistance",
}

var decisionPhrases = []string{
	"agreed",
	"decided",
	"decision",
	"promised",
	"remind",
	"reminder",
	"settled",
	"will remind",
}

func ValidateMemoryDocument(content string, limit int) error {
	if err := validateBase(content, limit); err != nil {
		return err
	}
	if isLowValueMemory(content) {
		return fmt.Errorf("memory contains transient or generic interaction boilerplate")
	}
	return nil
}

func ValidateProfileDocument(content string, limit int) error {
	if err := validateBase(content, limit); err != nil {
		return err
	}
	if isGenericProfile(content) {
		return fmt.Errorf("profile contains generic assistant boilerplate")
	}
	return nil
}

func ValidateUserDocument(content string, limit int) error {
	if err := validateBase(content, limit); err != nil {
		return err
	}
	if isLowValueMemory(content) {
		return fmt.Errorf("user profile contains transient or generic interaction boilerplate")
	}
	return nil
}

func NormalizeTaggedMemory(memoryDoc, userDoc string) (string, string) {
	memoryLines := normalizeTaggedLines(memoryDoc, true)
	userLines := normalizeTaggedLines(userDoc, false)

	filteredUser := make([]string, 0, len(userLines))
	for _, line := range userLines {
		if lineTag(line) == "<decision>" {
			memoryLines = append(memoryLines, line)
			continue
		}
		filteredUser = append(filteredUser, line)
	}

	return strings.Join(dedupeLines(memoryLines), "\n"), strings.Join(dedupeLines(filteredUser), "\n")
}

func validateBase(content string, limit int) error {
	if len(content) > limit {
		return fmt.Errorf("document exceeds limit: %d > %d", len(content), limit)
	}

	lower := strings.ToLower(content)
	for _, char := range invisibleChars {
		if strings.Contains(lower, char) {
			return fmt.Errorf("document contains invisible unicode character")
		}
	}

	for _, threat := range threatPatterns {
		if threat.re.MatchString(lower) {
			return fmt.Errorf("document matches blocked pattern %q", threat.label)
		}
	}

	return nil
}

func isLowValueMemory(content string) bool {
	lines := significantLines(content)
	if len(lines) == 0 {
		return false
	}

	flagged := 0
	for _, line := range lines {
		if containsAny(line, transientMemoryPhrases) || containsAny(line, genericProfilePhrases) {
			flagged++
		}
	}
	return flagged == len(lines)
}

func isGenericProfile(content string) bool {
	lines := significantLines(content)
	if len(lines) == 0 {
		return false
	}

	flagged := 0
	for _, line := range lines {
		if containsAny(line, genericProfilePhrases) {
			flagged++
		}
	}
	if flagged == 0 {
		return false
	}
	return flagged == len(lines)
}

func significantLines(content string) []string {
	lines := strings.Split(content, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") {
			line = strings.TrimSpace(line[2:])
		}
		out = append(out, strings.ToLower(line))
	}
	return out
}

func containsAny(content string, phrases []string) bool {
	for _, phrase := range phrases {
		if strings.Contains(content, phrase) {
			return true
		}
	}
	return false
}

func normalizeTaggedLines(doc string, allowDecision bool) []string {
	lines := strings.Split(strings.TrimSpace(doc), "\n")
	out := make([]string, 0, len(lines))
	for _, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "- ") || strings.HasPrefix(line, "* ") {
			line = strings.TrimSpace(line[2:])
		}
		content := strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(line, "<memory>"), "<decision>"))
		isDecision := looksLikeDecision(content)
		switch {
		case allowDecision && isDecision:
			line = "<decision> " + content
		case !allowDecision && isDecision:
			line = "<decision> " + content
		default:
			line = "<memory> " + content
		}
		out = append(out, line)
	}
	return out
}

func looksLikeDecision(line string) bool {
	return containsAny(strings.ToLower(line), decisionPhrases)
}

func lineTag(line string) string {
	for _, tag := range []string{"<memory>", "<decision>"} {
		if strings.HasPrefix(line, tag) {
			return tag
		}
	}
	return ""
}

func dedupeLines(lines []string) []string {
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		if slices.Contains(out, line) {
			continue
		}
		out = append(out, line)
	}
	return out
}
