// Script to execute the migration SQL files directly against Supabase
const fs = require('fs');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

// Supabase connection details
const supabaseUrl = "https://oizzdexnvcquljbeogwr.supabase.co";
const supabaseServiceKey = process.env.SUPABASE_SERVICE_KEY || ""; // Use the service key, not the anon key

// Create a Supabase client with the service key
const supabase = createClient(supabaseUrl, supabaseServiceKey);

async function applyMigration() {
  if (!supabaseServiceKey) {
    console.error("Error: SUPABASE_SERVICE_KEY environment variable is not set");
    console.log("Please set it using: $env:SUPABASE_SERVICE_KEY = 'your-service-key'");
    process.exit(1);
  }

  try {
    // Get the migration file
    const migrationFile = path.join(__dirname, '../supabase/migrations/20230825000001_dashboard_statistics.sql');
    
    if (!fs.existsSync(migrationFile)) {
      console.error(`Migration file not found: ${migrationFile}`);
      process.exit(1);
    }
    
    // Read the SQL content
    const sql = fs.readFileSync(migrationFile, 'utf-8');
    
    console.log("Applying migration...");
    
    // Execute the SQL
    const { data, error } = await supabase.rpc('exec_sql', { sql_query: sql });
    
    if (error) {
      console.error("Error applying migration:", error);
      process.exit(1);
    }
    
    console.log("Migration applied successfully!");
    console.log(data);
    
  } catch (error) {
    console.error("Error:", error);
    process.exit(1);
  }
}

// Run the function
applyMigration(); 