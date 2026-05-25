# Screenshot capture

Requires the dev server running (`npm run dev`) and valid Supabase credentials.

```powershell
$env:SCREENSHOT_EMAIL="your@email.com"
$env:SCREENSHOT_PASSWORD="your-password"
$env:APP_URL="http://localhost:3001"
npm run screenshots
```

Output: `public/images/screenshots/*.png`

Do not commit passwords. Use environment variables only.
