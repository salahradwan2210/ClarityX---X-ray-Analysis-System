"use client"

import type React from "react"

import { createContext, useContext, useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import type { Session, User } from "@supabase/supabase-js"
import { supabase } from "@/lib/supabase"

type Profile = {
  id: string
  full_name: string
  phone_number: string
  specialty: string
  hospital: string
}

type AuthContextType = {
  user: User | null
  session: Session | null
  profile: Profile | null
  isLoading: boolean
  signUp: (email: string, password: string, userData: Omit<Profile, "id">) => Promise<void>
  signIn: (email: string, password: string) => Promise<void>
  signOut: () => Promise<void>
  refreshProfile: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

// Create mock user for development - NO LONGER USED
// const mockUser = {
//   id: "mock-user-id",
//   email: "doctor@example.com",
//   app_metadata: {},
//   user_metadata: {},
//   aud: "authenticated",
//   created_at: new Date().toISOString(),
// } as User

// const mockSession = {
//   access_token: "mock-access-token",
//   refresh_token: "mock-refresh-token",
//   user: mockUser,
//   expires_at: Date.now() + 3600,
// } as Session

// const mockProfile = {
//   id: "mock-user-id",
//   full_name: "Dr. John Doe",
//   phone_number: "+1234567890",
//   specialty: "Radiologist",
//   hospital: "Cairo Medical Center",
// }

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [profile, setProfile] = useState<Profile | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()

  const refreshProfile = async () => {
    if (!user) return

    try {
      console.log("Fetching profile for user:", user.id);
      
      // Always use real profile data
      const { data, error } = await supabase.from("profiles").select("*").eq("id", user.id).single()

      if (error) {
        console.error("Error fetching profile - Details:", JSON.stringify(error));
        
        // If the error is empty or related to not finding the record, create a new profile
        if (JSON.stringify(error) === '{}' || error.code === 'PGRST116' || error.message?.includes('not found')) {
          console.log("Creating default profile for user:", user.id);
          
          // Create a default profile
          try {
            const { data: newProfile, error: insertError } = await supabase
              .from("profiles")
              .insert([{
                id: user.id,
                full_name: user.user_metadata?.full_name || "",
                phone_number: user.user_metadata?.phone_number || "",
                specialty: "",
                hospital: ""
              }])
              .select();
              
            if (insertError) {
              console.error("Error creating default profile:", insertError);
              // Continue with login despite error - use empty profile
              setProfile(null);
              return;
            }
            
            console.log("Created new profile successfully:", newProfile);
            setProfile(newProfile?.[0] || null);
            return;
          } catch (insertCatchError) {
            console.error("Exception while creating profile:", insertCatchError);
            // Continue with login despite error
            setProfile(null);
            return;
          }
        }
        
        // For other errors, continue with null profile
        setProfile(null);
        return;
      }

      if (!data) {
        console.log("No profile data found, creating default profile");
        
        // Create a default profile
        try {
          const { data: newProfile, error: insertError } = await supabase
            .from("profiles")
            .insert([{
              id: user.id,
              full_name: user.user_metadata?.full_name || "",
              phone_number: user.user_metadata?.phone_number || "",
              specialty: "",
              hospital: ""
            }])
            .select();
            
          if (insertError) {
            console.error("Error creating default profile:", insertError);
            // Continue with login despite error
            setProfile(null);
            return;
          }
          
          console.log("Created new profile successfully:", newProfile);
          setProfile(newProfile?.[0] || null);
          return;
        } catch (insertCatchError) {
          console.error("Exception while creating profile:", insertCatchError);
          // Continue with login despite error
          setProfile(null);
          return;
        }
      }

      console.log("Profile found successfully:", data);
      setProfile(data)
    } catch (error) {
      console.error("Unhandled error in refreshProfile:", error);
      // Always continue with login process, even with a null profile
      setProfile(null);
    }
  }

  useEffect(() => {
    const setData = async () => {
      try {
        // Always use real authentication
        const {
          data: { session },
          error,
        } = await supabase.auth.getSession()
        if (error) {
          console.error(error)
          setIsLoading(false)
          return
        }

        setSession(session)
        setUser(session?.user ?? null)

        if (session?.user) {
          await refreshProfile()
        }

        setIsLoading(false)
      } catch (error) {
        console.error("Auth initialization error:", error);
        setIsLoading(false);
      }
    }

    const { data: authListener } = supabase.auth.onAuthStateChange(async (event, session) => {
      setSession(session)
      setUser(session?.user ?? null)

      if (session?.user) {
        await refreshProfile()
      } else {
        setProfile(null)
      }

      setIsLoading(false)
    })

    setData()

    return () => {
      authListener.subscription.unsubscribe()
    }
  }, [])

  const signUp = async (
    email: string,
    password: string,
    userData: Omit<Profile, "id"> = { full_name: "", phone_number: "", specialty: "", hospital: "" },
  ) => {
    setIsLoading(true)
    
    // Always use real signup
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: userData.full_name,
          phone_number: userData.phone_number,
        },
      },
    })

    if (error) {
      setIsLoading(false);
      throw error;
    }

    if (data.user) {
      // Create profile record
      const { error: profileError } = await supabase.from("profiles").insert([
        {
          id: data.user.id,
          ...userData,
        },
      ])

      if (profileError) {
        console.error("Error creating profile:", profileError)
        // Continue anyway as the auth user was created
      }
    }

    setIsLoading(false);
    router.push("/login?registered=true")
  }

  const signIn = async (email: string, password: string) => {
    setIsLoading(true)
    
    try {
      console.log("Attempting sign in for:", email);
      
      // Always use real signin
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })

      if (error) {
        console.error("Sign in authentication error:", error);
        setIsLoading(false);
        throw error;
      }

      console.log("Authentication successful for user:", data.user?.id);
      setUser(data.user);
      setSession(data.session);
      
      // Try to get profile but don't block login if it fails
      try {
        await refreshProfile();
      } catch (profileError) {
        console.error("Error refreshing profile, but continuing with login:", profileError);
        // If profile is null at this point, create a minimal profile object
        if (!profile) {
          setProfile({
            id: data.user.id,
            full_name: data.user.user_metadata?.full_name || "",
            phone_number: data.user.user_metadata?.phone_number || "",
            specialty: "",
            hospital: ""
          });
        }
      }
      
      setIsLoading(false);
      router.push("/dashboard");
    } catch (error) {
      console.error("Complete sign in error:", error);
      setIsLoading(false);
      throw error;
    }
  }

  const signOut = async () => {
    setIsLoading(true)
    
    // Always use real signout
    const { error } = await supabase.auth.signOut()

    if (error) {
      setIsLoading(false);
      throw error;
    }

    setUser(null);
    setSession(null);
    setProfile(null);
    setIsLoading(false);
    router.push("/login");
  }

  const value = {
    user,
    session,
    profile,
    isLoading,
    signUp,
    signIn,
    signOut,
    refreshProfile,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
