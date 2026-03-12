package storage

import (
	"database/sql"
	"fmt"
	"regexp"
	"strings"
	"time"

	_ "modernc.org/sqlite"
)

type SessionStore struct {
	db *sql.DB
}

type Message struct {
	ID        int64
	SessionID string
	UserID    string
	Role      string
	Content   string
	CreatedAt time.Time
}

var ftsTokenPattern = regexp.MustCompile(`[\pL\pN]+`)
var ftsStopWords = map[string]struct{}{
	"a": {}, "an": {}, "and": {}, "before": {}, "did": {}, "do": {}, "does": {},
	"i": {}, "me": {}, "my": {}, "remember": {}, "tell": {}, "told": {}, "we": {},
	"what": {}, "you": {},
}

const schemaSQL = `
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  started_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_created_at ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  content,
  content='messages',
  content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
`

func NewSessionStore(path string) (*SessionStore, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(`PRAGMA busy_timeout = 3000;`); err != nil {
		_ = db.Close()
		return nil, err
	}
	return &SessionStore{db: db}, nil
}

func (s *SessionStore) Close() error {
	if s.db == nil {
		return nil
	}
	return s.db.Close()
}

func (s *SessionStore) EnsureSchema() error {
	_, err := s.db.Exec(schemaSQL)
	return err
}

func (s *SessionStore) EnsureSessionForUser(sessionID, userID string) error {
	if strings.TrimSpace(userID) == "" {
		return fmt.Errorf("user_id is required")
	}
	if _, err := s.db.Exec(
		`INSERT OR IGNORE INTO sessions(id, user_id, started_at) VALUES(?, ?, ?)`,
		sessionID,
		userID,
		time.Now().UTC().Format(time.RFC3339Nano),
	); err != nil {
		return err
	}

	var existingUserID string
	if err := s.db.QueryRow(`SELECT user_id FROM sessions WHERE id = ?`, sessionID).Scan(&existingUserID); err != nil {
		return err
	}
	if existingUserID != userID {
		return fmt.Errorf("session %q belongs to user %q, not %q", sessionID, existingUserID, userID)
	}
	return nil
}

func (s *SessionStore) AppendMessage(sessionID, role, content string, createdAt time.Time) error {
	_, err := s.db.Exec(
		`INSERT INTO messages(session_id, role, content, created_at) VALUES(?, ?, ?, ?)`,
		sessionID,
		role,
		content,
		createdAt.UTC().Format(time.RFC3339Nano),
	)
	return err
}

func (s *SessionStore) RecentMessages(sessionID string, limit int) ([]Message, error) {
	rows, err := s.db.Query(
		`SELECT m.id, m.session_id, se.user_id, m.role, m.content, m.created_at
		 FROM messages m
		 JOIN sessions se ON se.id = m.session_id
		 WHERE m.session_id = ?
		 ORDER BY m.created_at DESC
		 LIMIT ?`,
		sessionID,
		limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []Message
	for rows.Next() {
		message, err := scanMessage(rows)
		if err != nil {
			return nil, err
		}
		messages = append(messages, message)
	}
	reverse(messages)
	return messages, rows.Err()
}

func (s *SessionStore) MessagesBetweenForUser(userID string, start, end time.Time, limit int) ([]Message, error) {
	rows, err := s.db.Query(
		`SELECT m.id, m.session_id, se.user_id, m.role, m.content, m.created_at
		 FROM messages m
		 JOIN sessions se ON se.id = m.session_id
		 WHERE se.user_id = ? AND m.created_at >= ? AND m.created_at <= ?
		 ORDER BY m.created_at ASC
		 LIMIT ?`,
		userID,
		start.UTC().Format(time.RFC3339Nano),
		end.UTC().Format(time.RFC3339Nano),
		limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []Message
	for rows.Next() {
		message, err := scanMessage(rows)
		if err != nil {
			return nil, err
		}
		messages = append(messages, message)
	}
	return messages, rows.Err()
}

func (s *SessionStore) SearchMessages(query, currentSessionID, userID string, limit int) ([]Message, error) {
	query = buildFTSQuery(query)
	if query == "" {
		return nil, nil
	}

	rows, err := s.db.Query(
		`SELECT m.id, m.session_id, se.user_id, m.role, m.content, m.created_at
		 FROM messages_fts f
		 JOIN messages m ON m.id = f.rowid
		 JOIN sessions se ON se.id = m.session_id
		 WHERE messages_fts MATCH ?
		   AND m.session_id != ?
		   AND se.user_id = ?
		 ORDER BY bm25(messages_fts)
		 LIMIT ?`,
		query,
		currentSessionID,
		userID,
		limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []Message
	for rows.Next() {
		message, err := scanMessage(rows)
		if err != nil {
			return nil, err
		}
		messages = append(messages, message)
	}
	return messages, rows.Err()
}

func (s *SessionStore) RecentMessagesAcrossAgent(limit int) ([]Message, error) {
	rows, err := s.db.Query(
		`SELECT m.id, m.session_id, se.user_id, m.role, m.content, m.created_at
		 FROM messages m
		 JOIN sessions se ON se.id = m.session_id
		 ORDER BY m.created_at DESC
		 LIMIT ?`,
		limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []Message
	for rows.Next() {
		message, err := scanMessage(rows)
		if err != nil {
			return nil, err
		}
		messages = append(messages, message)
	}
	reverse(messages)
	return messages, rows.Err()
}

func scanMessage(scanner interface {
	Scan(dest ...any) error
}) (Message, error) {
	var (
		message   Message
		createdAt string
	)
	if err := scanner.Scan(&message.ID, &message.SessionID, &message.UserID, &message.Role, &message.Content, &createdAt); err != nil {
		return Message{}, err
	}
	parsed, err := time.Parse(time.RFC3339Nano, createdAt)
	if err != nil {
		return Message{}, fmt.Errorf("parse message time: %w", err)
	}
	message.CreatedAt = parsed
	return message, nil
}

func reverse(messages []Message) {
	for left, right := 0, len(messages)-1; left < right; left, right = left+1, right-1 {
		messages[left], messages[right] = messages[right], messages[left]
	}
}

func buildFTSQuery(query string) string {
	query = strings.TrimSpace(query)
	if query == "" {
		return ""
	}

	// Build a plain token query so user punctuation and FTS operators cannot break MATCH syntax.
	tokens := ftsTokenPattern.FindAllString(query, -1)
	filtered := make([]string, 0, len(tokens))
	for _, token := range tokens {
		if _, blocked := ftsStopWords[strings.ToLower(token)]; blocked {
			continue
		}
		filtered = append(filtered, token)
	}
	if len(filtered) == 0 {
		filtered = tokens
	}
	if len(filtered) == 0 {
		return ""
	}
	for i, token := range filtered {
		filtered[i] = `"` + token + `"`
	}
	return strings.Join(filtered, " ")
}
