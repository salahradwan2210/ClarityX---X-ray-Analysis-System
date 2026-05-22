// Script to execute SQL queries directly against Supabase using the JS client
const { createClient } = require('@supabase/supabase-js');
const fs = require('fs');
const path = require('path');

// Read the migration SQL file
const migrationPath = path.join(__dirname, '../supabase/migrations/20230825000001_dashboard_statistics.sql');
const migrationSQL = fs.readFileSync(migrationPath, 'utf8');

// Supabase connection
const supabaseUrl = 'https://oizzdexnvcquljbeogwr.supabase.co';
const supabaseKey = process.env.SUPABASE_KEY || '';
const supabase = createClient(supabaseUrl, supabaseKey);

async function executeMigrations() {
  if (!supabaseKey) {
    console.error('SUPABASE_KEY environment variable is not set.');
    console.error('Set it with: $env:SUPABASE_KEY="your-service-role-key"');
    process.exit(1);
  }

  try {
    console.log('Executing migrations...');
    
    // Split the SQL by the function definitions
    const functionDefinitions = migrationSQL.split('CREATE OR REPLACE FUNCTION');
    
    // Skip the first element which is empty or just comments
    for (let i = 1; i < functionDefinitions.length; i++) {
      const functionDef = 'CREATE OR REPLACE FUNCTION' + functionDefinitions[i];
      
      console.log(`Executing function definition ${i}...`);
      
      const { error } = await supabase.rpc('exec_sql', { sql: functionDef });
      
      if (error) {
        console.error(`Error executing function definition ${i}:`, error);
      } else {
        console.log(`Function ${i} executed successfully.`);
      }
    }
    
    console.log('All migrations completed.');
    
  } catch (error) {
    console.error('Error executing migrations:', error);
  }
}

// Run the script
executeMigrations().catch(console.error); 