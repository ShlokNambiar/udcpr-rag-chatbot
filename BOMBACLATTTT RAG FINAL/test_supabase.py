from supabase_config import initialize_supabase, create_chat_session, save_message, get_chat_history

# Initialize Supabase
supabase = initialize_supabase()
if not supabase:
    print("Failed to initialize Supabase")
    exit(1)

# Use existing session ID
session_id = 'c580bfc0-3438-47dc-bf5a-3a3b45638600'
print(f"Using session: {session_id}")

# Save test messages
save_message(supabase, session_id, 'user', 'Hello, this is a test message')
save_message(supabase, session_id, 'assistant', 'This is a test response')
print("Saved test messages")

# Retrieve messages
messages = get_chat_history(supabase, session_id)
print(f"Retrieved {len(messages)} messages:")
for msg in messages:
    print(f"{msg['role']}: {msg['content']}")
