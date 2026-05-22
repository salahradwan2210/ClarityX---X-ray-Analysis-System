# Figure 5.1: DiagnoLink System Technical Architecture Diagram

```
+-----------------------+          +-------------------------+          +------------------------+
|                       |  HTTP/   |                         |  SQL/    |                        |
|    FRONTEND (Client)  | WebSocket|     BACKEND (Server)    |  REST    |       DATABASE         |
|                       +--------->|                         +--------->|                        |
+-----------------------+          +-------------------------+          +------------------------+
| • Next.js 15.2.4      |          | • Python FastAPI        |          | • Supabase             |
| • React/TypeScript    |          | • AI Model Service      |          | • PostgreSQL           |
| • Tailwind CSS        |   API    | • Image Processing      |   API    | • Authentication       |
| • React Hooks         | Requests | • ConvNeXt Large CNN    |  Calls   | • Storage Service      |
| • Framer Motion       |          | • Prediction Pipeline   |          | • JWT Management       |
| • React-PDF           |          | • Localization Engine   |          | • Role-based Access    |
+-----------------------+          +-------------------------+          +------------------------+
       ^                                      ^                                   ^
       |                                      |                                   |
       |           +---------------------------------------------+               |
       |           |                                             |               |
       +-----------|              USER INTERACTIONS              |---------------+
                   |                                             |
                   +---------------------------------------------+
                   | • Image Upload                              |
                   | • Patient Management                        |
                   | • Analysis Requests                         |
                   | • Report Generation                         |
                   | • Authentication Flows                      |
                   +---------------------------------------------+

```

## Diagram Description

This technical architecture diagram illustrates the three-tier architecture of the DiagnoLink system:

1. **Frontend (Client)**
   - Built with Next.js 15.2.4 and React
   - Uses TypeScript for type safety
   - Styled with Tailwind CSS
   - Uses React Hooks for state management
   - Integrates Framer Motion for animations
   - Includes React-PDF for report generation

2. **Backend (Server)**
   - Python-based FastAPI application
   - Hosts the ConvNeXt Large CNN model
   - Handles image processing and analysis
   - Provides prediction and localization services
   - Exposes RESTful API endpoints

3. **Database**
   - Supabase cloud database platform
   - PostgreSQL database for data persistence
   - Authentication and user management
   - Secure storage services
   - JWT-based security

The arrows indicate data flow and communication between components. The frontend communicates with the backend via HTTP/WebSocket API requests, while the backend interacts with the database through SQL queries and REST API calls. User interactions trigger various flows that involve all three tiers of the architecture.

**Note:** This ASCII representation can be used as a reference to create a professional diagram using tools like Draw.io, Lucidchart, or Microsoft Visio. 