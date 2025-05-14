-- Supabase SQL Schema for Chat Memory

-- Chat Sessions Table
CREATE TABLE chat_memories (
    session_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'anonymous',
    title TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Chat Messages Table
CREATE TABLE chat_messages (
    id UUID PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES chat_memories(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    metadata JSONB,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Create indexes for faster queries
CREATE INDEX idx_chat_memories_user_id ON chat_memories(user_id);
CREATE INDEX idx_chat_memories_updated_at ON chat_memories(updated_at DESC);

CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_timestamp ON chat_messages(timestamp);
CREATE INDEX idx_chat_messages_role ON chat_messages(role);

-- Enable Realtime for chat messages
BEGIN;
  DROP PUBLICATION IF EXISTS supabase_realtime;
  CREATE PUBLICATION supabase_realtime;
COMMIT;
ALTER PUBLICATION supabase_realtime ADD TABLE chat_messages;
ALTER PUBLICATION supabase_realtime ADD TABLE chat_memories;

-- Row Level Security Policies
-- Enable RLS
ALTER TABLE chat_memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust these based on your authentication setup)
-- For anonymous access, we'll allow all operations
CREATE POLICY "Allow anonymous access to chat_memories"
    ON chat_memories FOR ALL
    USING (true);

CREATE POLICY "Allow anonymous access to chat_messages"
    ON chat_messages FOR ALL
    USING (true);

-- If you implement authentication later, you can use policies like:
-- CREATE POLICY "Users can only access their own chat sessions"
--    ON chat_memories FOR ALL
--    USING (auth.uid()::text = user_id);

-- CREATE POLICY "Users can only access messages from their sessions"
--    ON chat_messages FOR ALL
--    USING (
--        session_id IN (
--            SELECT session_id FROM chat_memories WHERE user_id = auth.uid()::text
--        )
--    );

-- Function to update the updated_at timestamp when a message is added
CREATE OR REPLACE FUNCTION update_chat_memory_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE chat_memories
    SET updated_at = NOW()
    WHERE session_id = NEW.session_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to update the chat_memories updated_at timestamp
CREATE TRIGGER update_chat_memory_timestamp_trigger
AFTER INSERT ON chat_messages
FOR EACH ROW
EXECUTE FUNCTION update_chat_memory_timestamp();
