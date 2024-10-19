import React from 'react'
import {BrowserRouter , Route, Routes} from 'react-router-dom'
import Navbar from './components/Navbar'
import FirstTop from './components/FirstTop'
import TopFooter from './components/TopFooter'

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <FirstTop />
      <TopFooter />
      <Routes>

      </Routes>
    </BrowserRouter>
  )
}

export default App