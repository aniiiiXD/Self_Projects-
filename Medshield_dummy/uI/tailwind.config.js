/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      keyframes: {
        borderAnimation: {
          '0%': { width: '0%', height: '0%', borderWidth: '0' },
          '25%': { width: '100%', height: '0%', borderWidth: '4px' },
          '50%': { width: '100%', height: '100%', borderWidth: '4px' },
          '100%': { width: '100%', height: '100%', borderWidth: '4px' },
        },
      },
      animation: {
        'border-draw': 'borderAnimation 2s ease-in-out forwards',
      },
    },
  },
  plugins: [],
}