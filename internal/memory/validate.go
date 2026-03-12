package memory

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"github.com/ErniConcepts/mynah/internal/llm"
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

var userScopedPhrases = []string{
	"user",
	"user's name",
	"name is",
	"prefers",
	"likes ",
	"works best for",
	"communication style",
	"answer me",
	"answers",
	"replies",
}

var sharedMemoryPhrases = []string{
	"barn",
	"horse",
	"gate",
	"routine",
	"reminder",
	"schedule",
	"company",
	"visitor",
	"employee",
	"front desk",
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

func ApplyMemoryOperations(memoryDoc, userDoc, userID string, operations []llm.MemoryOperation) (string, string, error) {
	memoryEntries := documentLines(memoryDoc)
	userEntries := documentLines(userDoc)

	for _, operation := range operations {
		target := strings.ToLower(strings.TrimSpace(operation.Target))
		action := strings.ToLower(strings.TrimSpace(operation.Action))
		content := strings.TrimSpace(operation.Content)
		oldText := strings.TrimSpace(operation.OldText)

		if target != "memory" && target != "user" {
			return "", "", fmt.Errorf("invalid memory operation target %q", operation.Target)
		}
		if action != "add" && action != "replace" && action != "remove" {
			return "", "", fmt.Errorf("invalid memory operation action %q", operation.Action)
		}
		if action == "add" && content == "" {
			return "", "", fmt.Errorf("add operation requires content")
		}
		if action == "replace" && (content == "" || oldText == "") {
			return "", "", fmt.Errorf("replace operation requires content and old_text")
		}
		if action == "remove" && oldText == "" {
			return "", "", fmt.Errorf("remove operation requires old_text")
		}
		if content != "" {
			if err := validateOperationContent(target, content, userID); err != nil {
				return "", "", err
			}
		}

		entries := &memoryEntries
		if target == "user" {
			entries = &userEntries
		}

		switch action {
		case "add":
			for _, atomicEntry := range splitAtomicEntries(content) {
				*entries = appendUniqueEntry(*entries, atomicEntry)
			}
		case "replace":
			nextEntries, err := replaceEntry(*entries, oldText, splitAtomicEntries(content))
			if err != nil {
				return "", "", err
			}
			*entries = nextEntries
		case "remove":
			nextEntries, err := removeEntry(*entries, oldText)
			if err != nil {
				return "", "", err
			}
			*entries = nextEntries
		}
	}

	return renderDocument(memoryEntries), renderDocument(userEntries), nil
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

func documentLines(content string) []string {
	raw := strings.Split(content, "\n")
	lines := make([]string, 0, len(raw))
	for _, line := range raw {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "- ")
		line = strings.TrimPrefix(line, "* ")
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		lines = append(lines, line)
	}
	return lines
}

func isUserScopedLine(line, userID, userName string) bool {
	lower := canonicalLine(line)
	if lower == "" {
		return false
	}
	if containsAny(lower, userScopedPhrases) {
		return true
	}
	if userName != "" && strings.Contains(lower, strings.ToLower(userName)) {
		return true
	}
	if userID != "" && strings.Contains(lower, strings.ToLower(userID)) {
		return true
	}
	return false
}

func isSharedLine(line, userID, userName string) bool {
	lower := canonicalLine(line)
	if lower == "" {
		return false
	}
	if containsAny(lower, sharedMemoryPhrases) {
		return true
	}
	if userName != "" && !strings.Contains(lower, strings.ToLower(userName)) && containsAny(lower, []string{"reminder", "routine", "schedule"}) {
		return true
	}
	if userID != "" && !strings.Contains(lower, strings.ToLower(userID)) && containsAny(lower, []string{"reminder", "routine", "schedule"}) {
		return true
	}
	return false
}

func referencesDifferentUser(line, userID, userName string) bool {
	lower := canonicalLine(line)
	if lower == "" || !isUserScopedLine(line, userID, userName) {
		return false
	}

	current := []string{}
	if trimmed := strings.ToLower(strings.TrimSpace(userID)); trimmed != "" {
		current = append(current, trimmed)
	}
	if trimmed := strings.ToLower(strings.TrimSpace(userName)); trimmed != "" && trimmed == strings.ToLower(strings.TrimSpace(userID)) {
		current = append(current, trimmed)
	}

	for _, token := range extractUserMarkers(lower) {
		if token == "" {
			continue
		}
		matchesCurrent := false
		for _, allowed := range current {
			if token == allowed {
				matchesCurrent = true
				break
			}
		}
		if !matchesCurrent {
			return true
		}
	}
	return false
}

func extractUserMarkers(lower string) []string {
	markers := []string{}
	patterns := []string{"user's name is ", "user name is ", "name is "}
	for _, pattern := range patterns {
		if idx := strings.Index(lower, pattern); idx >= 0 {
			value := strings.TrimSpace(lower[idx+len(pattern):])
			value = strings.Trim(value, " .,:;!?")
			if value != "" {
				markers = append(markers, strings.Trim(strings.Fields(value)[0], " .,:;!?"))
			}
		}
	}

	words := strings.Fields(lower)
	if len(words) >= 2 && containsAny(lower, []string{"prefers", "likes", "answers", "replies"}) {
		first := strings.Trim(words[0], " .,:;!?")
		if first != "" && !strings.HasPrefix(first, "user") && first != "prefers" {
			markers = append(markers, first)
		}
	}
	return markers
}

func canonicalLine(line string) string {
	return strings.ToLower(strings.TrimSpace(strings.TrimSuffix(line, ".")))
}

func bulletLine(line string) string {
	line = strings.TrimSpace(line)
	if line == "" {
		return ""
	}
	return "- " + line
}

func validateOperationContent(target, content, userID string) error {
	if err := validateBase(content, 100000); err != nil {
		return err
	}

	if target == "user" {
		if referencesDifferentUser(content, userID, userID) {
			return fmt.Errorf("user memory operation references a different user")
		}
		if !isUserScopedLine(content, userID, userID) {
			return fmt.Errorf("user memory operation requires user-scoped content")
		}
		return nil
	}

	if referencesDifferentUser(content, userID, userID) {
		return fmt.Errorf("shared memory operation references a different user")
	}
	if isUserScopedLine(content, userID, userID) && !isSharedLine(content, userID, userID) {
		return fmt.Errorf("shared memory operation cannot store user-scoped content")
	}
	return nil
}

func appendUniqueEntry(entries []string, content string) []string {
	key := canonicalLine(content)
	for _, entry := range entries {
		if canonicalLine(entry) == key {
			return entries
		}
	}
	return append(entries, content)
}

func replaceEntry(entries []string, oldText string, contentEntries []string) ([]string, error) {
	matches := matchingEntryIndexes(entries, oldText)
	if len(matches) == 0 {
		return nil, fmt.Errorf("no entry matched %q", oldText)
	}
	if len(matches) > 1 && !allMatchingEntriesIdentical(entries, matches) {
		return nil, fmt.Errorf("multiple entries matched %q", oldText)
	}
	if len(contentEntries) == 0 {
		return nil, fmt.Errorf("replace operation requires replacement content")
	}
	index := matches[0]
	next := make([]string, 0, len(entries)-1+len(contentEntries))
	next = append(next, entries[:index]...)
	next = append(next, contentEntries...)
	next = append(next, entries[index+1:]...)
	return dedupeEntries(next), nil
}

func removeEntry(entries []string, oldText string) ([]string, error) {
	matches := matchingEntryIndexes(entries, oldText)
	if len(matches) == 0 {
		return nil, fmt.Errorf("no entry matched %q", oldText)
	}
	if len(matches) > 1 && !allMatchingEntriesIdentical(entries, matches) {
		return nil, fmt.Errorf("multiple entries matched %q", oldText)
	}
	index := matches[0]
	return append(entries[:index], entries[index+1:]...), nil
}

func matchingEntryIndexes(entries []string, oldText string) []int {
	lowerNeedle := canonicalLine(strings.TrimSpace(strings.TrimLeft(oldText, "-* ")))
	matches := make([]int, 0, len(entries))
	for index, entry := range entries {
		if lowerNeedle != "" && strings.Contains(canonicalLine(entry), lowerNeedle) {
			matches = append(matches, index)
		}
	}
	return matches
}

func allMatchingEntriesIdentical(entries []string, indexes []int) bool {
	if len(indexes) <= 1 {
		return true
	}
	first := canonicalLine(entries[indexes[0]])
	for _, index := range indexes[1:] {
		if canonicalLine(entries[index]) != first {
			return false
		}
	}
	return true
}

func dedupeEntries(entries []string) []string {
	out := make([]string, 0, len(entries))
	seen := map[string]struct{}{}
	for _, entry := range entries {
		key := canonicalLine(entry)
		if key == "" {
			continue
		}
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, strings.TrimSpace(entry))
	}
	return out
}

func renderDocument(entries []string) string {
	if len(entries) == 0 {
		return ""
	}
	lines := make([]string, 0, len(entries))
	for _, entry := range dedupeEntries(entries) {
		lines = append(lines, bulletLine(entry))
	}
	return strings.Join(lines, "\n")
}

func splitAtomicEntries(content string) []string {
	content = strings.TrimSpace(content)
	if content == "" {
		return nil
	}

	parts := make([]string, 0, 4)
	var current strings.Builder
	flush := func() {
		value := strings.TrimSpace(current.String())
		value = strings.TrimLeft(value, "-* ")
		value = strings.TrimSpace(value)
		if value != "" {
			parts = append(parts, value)
		}
		current.Reset()
	}

	for _, r := range content {
		switch {
		case r == '\n' || r == ';':
			flush()
		default:
			current.WriteRune(r)
			if r == '.' || r == '!' || r == '?' {
				flush()
			}
		}
	}
	flush()

	if len(parts) <= 1 {
		return parts
	}

	normalized := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		last := rune(0)
		for _, r := range part {
			last = r
		}
		if last != 0 && !unicode.IsPunct(last) {
			part += "."
		}
		normalized = append(normalized, part)
	}
	return normalized
}
