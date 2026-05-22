import { NextRequest, NextResponse } from "next/server";

// This is a simplified middleware that doesn't use edge runtime
export function middleware(request: NextRequest) {
  // In development mode, always allow access
  if (process.env.NODE_ENV === 'development') {
    return NextResponse.next();
  }
  
  // For production, we'll do a simple check if user is logged in
  // This should be replaced with proper auth check in production
  const authCookie = request.cookies.get('auth-token');
  
  if (!authCookie) {
    // Redirect to login if no auth cookie
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("redirectedFrom", request.nextUrl.pathname);
    return NextResponse.redirect(url);
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: ["/dashboard", "/patients/:path*", "/analysis/:path*", "/results/:path*", "/reports/:path*"],
}; 