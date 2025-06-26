/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          light: '#667eea',
          DEFAULT: '#5a67d8',
          dark: '#4c51bf',
        },
        secondary: {
          light: '#764ba2',
          DEFAULT: '#6b46c1',
          dark: '#553c9a',
        },
        dark: {
          light: '#2d3748',
          DEFAULT: '#1a202c',
          dark: '#171923',
        },
      },
      backdropBlur: {
        xs: '2px',
      },
    },
  },
  plugins: [],
}