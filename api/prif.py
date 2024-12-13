from supabase import create_client, Client

url = "https://izhrccocgpcywgemzvmz.supabase.co"  # Replace with your Supabase URL
key = "yeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Iml6aHJjY29jZ3BjeXdnZW16dm16Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQxMTQ4MTQsImV4cCI6MjA0OTY5MDgxNH0.vPyORFp8dF2aL-ML5ZmxY4alHZO2q2FzB3Ga5EKgdGo"  # Replace with your API key

supabase: Client = create_client(url, key)

# Check if you can fetch data from a table
data = supabase.table("your_table").select("*").execute()
print(data)
