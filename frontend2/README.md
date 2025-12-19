# Goal Refinement Application

A Next.js web application for tracking performance, refining sales goals, and staying aligned with targets. Built with TypeScript and Tailwind CSS.

## Features

- **Responsive Login Page**: Professional login interface compatible with desktop and iPad displays
- **Role-based Authentication**: Supports FLM, SLM, and Home Office user roles
- **Modern UI**: Glass-morphism design with gradient backgrounds
- **TypeScript**: Full type safety and better development experience
- **Future-ready**: Structured for easy backend integration

## User Roles

- **FLM (Front Line Manager)**: Territory management and goal refinement
- **SLM (Second Line Manager)**: Territory and FLM goal oversight
- **Home Office**: Comprehensive analytics and oversight across all levels

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd goal
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

```
src/
├── app/
│   ├── dashboard/          # Role-based dashboard pages
│   │   ├── flm/           # FLM dashboard
│   │   ├── slm/           # SLM dashboard
│   │   └── home-office/   # Home Office dashboard
│   ├── login/             # Login page
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Home page (redirects to login)
├── components/
│   ├── auth/              # Authentication components
│   └── ui/                # UI components
├── lib/                   # Utility libraries
│   └── auth.ts           # Authentication service
└── types/                 # TypeScript type definitions
    └── auth.ts           # Authentication types
```

## Authentication

Currently uses mock authentication for development. The login form accepts any credentials and simulates different user roles.

### Test Credentials

- Any username/password combination will work
- The mock system randomly assigns user roles for testing

### Future Backend Integration

The authentication system is structured to easily integrate with a real backend:

1. Update `src/lib/auth.ts` with actual API endpoints
2. Replace mock responses with real API calls
3. Implement proper token management and refresh logic

## Responsive Design

The application is optimized for:
- **Desktop**: Full-featured layout with side-by-side login form
- **iPad**: Responsive design that adapts to tablet dimensions
- **Mobile**: Stacked layout for smaller screens

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server

### Styling

- Uses Tailwind CSS for styling
- Custom gradient backgrounds and glass-morphism effects
- Responsive design utilities for different screen sizes

## Design

The UI is based on Figma prototypes with:
- Custom gradient backgrounds
- Glass-morphism login form
- Professional color scheme
- Responsive typography

## Next Steps

1. Implement dashboard pages based on Figma designs
2. Integrate with backend authentication API
3. Add data fetching and state management
4. Implement role-based route protection
5. Add comprehensive error handling

## Contributing

1. Follow the existing code structure
2. Ensure TypeScript types are properly defined
3. Test responsive design on multiple screen sizes
4. Update documentation for new features