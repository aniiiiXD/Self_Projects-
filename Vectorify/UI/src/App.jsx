import { useState } from 'react'

import './App.css'
import Navbar from './component/Navbar'
import Title from './component/Title'
import UploadBar from './component/UploadBar'
import VerifyBar from './component/VerifyBar'
import KeysBar from './component/KeysBar'
import GitFlex from './component/GitFlex'

function App() {


  return (
    <>
      <Navbar />
      <div className='flex flex-col items-center justify-center min-h-screen'>
        <Title />
      </div>
      <UploadBar />
      <VerifyBar />
      <KeysBar />
      <GitFlex />
    </>
  )
} 

export default App
