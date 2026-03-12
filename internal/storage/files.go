package storage

import (
	"os"
	"path/filepath"
	"strings"
)

type AgentPaths struct {
	RootPath    string
	UsersPath   string
	MemoryPath  string
	ProfilePath string
	HistoryPath string
}

type FileStore struct {
	paths            AgentPaths
	memoryCharLimit  int
	profileCharLimit int
}

func NewAgentPaths(dataDir, tenantID, agentID string) AgentPaths {
	root := filepath.Join(dataDir, "tenants", tenantID, "agents", agentID)
	return AgentPaths{
		RootPath:    root,
		UsersPath:   filepath.Join(root, "users"),
		MemoryPath:  filepath.Join(root, "MEMORY.md"),
		ProfilePath: filepath.Join(root, "AGENT_PROFILE.md"),
		HistoryPath: filepath.Join(root, "history.db"),
	}
}

func EnsureAgentPaths(paths AgentPaths) error {
	if err := os.MkdirAll(paths.RootPath, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(paths.UsersPath, 0o755); err != nil {
		return err
	}
	if _, err := os.Stat(paths.MemoryPath); os.IsNotExist(err) {
		if err := os.WriteFile(paths.MemoryPath, []byte(""), 0o644); err != nil {
			return err
		}
	}
	if _, err := os.Stat(paths.ProfilePath); os.IsNotExist(err) {
		if err := os.WriteFile(paths.ProfilePath, []byte(""), 0o644); err != nil {
			return err
		}
	}
	return nil
}

func NewFileStore(paths AgentPaths, memoryCharLimit, profileCharLimit int) *FileStore {
	return &FileStore{
		paths:            paths,
		memoryCharLimit:  memoryCharLimit,
		profileCharLimit: profileCharLimit,
	}
}

func (s *FileStore) ReadMemory() (string, error) {
	return readTrimmed(s.paths.MemoryPath)
}

func (s *FileStore) ReadProfile() (string, error) {
	return readTrimmed(s.paths.ProfilePath)
}

func (s *FileStore) ReadUserProfile(userID string) (string, error) {
	return readTrimmed(s.userProfilePath(userID))
}

func (s *FileStore) WriteMemory(content string) error {
	return writeAtomically(s.paths.MemoryPath, strings.TrimSpace(content)+"\n")
}

func (s *FileStore) WriteProfile(content string) error {
	return writeAtomically(s.paths.ProfilePath, strings.TrimSpace(content)+"\n")
}

func (s *FileStore) WriteUserProfile(userID, content string) error {
	path := s.userProfilePath(userID)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return writeAtomically(path, strings.TrimSpace(content)+"\n")
}

func readTrimmed(path string) (string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", err
	}
	return strings.TrimSpace(string(raw)), nil
}

func (s *FileStore) userProfilePath(userID string) string {
	return filepath.Join(s.paths.UsersPath, userID, "USER.md")
}

func writeAtomically(path, content string) error {
	dir := filepath.Dir(path)
	temp, err := os.CreateTemp(dir, ".mynah-*")
	if err != nil {
		return err
	}
	tempPath := temp.Name()

	cleanup := func() {
		_ = os.Remove(tempPath)
	}

	if _, err := temp.WriteString(content); err != nil {
		_ = temp.Close()
		cleanup()
		return err
	}
	if err := temp.Close(); err != nil {
		cleanup()
		return err
	}
	if err := os.Chmod(tempPath, 0o644); err != nil {
		cleanup()
		return err
	}
	if err := os.Rename(tempPath, path); err != nil {
		cleanup()
		return err
	}
	return nil
}
