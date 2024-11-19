import { useState } from 'react'
import {BrowserRouter , Router , Routes} from 'react-router-dom';
import Navbar from './components/Navbar';

function App() {
  

  return (
    <BrowserRouter>
      <Navbar />
    </BrowserRouter>
  )
}

export default App
