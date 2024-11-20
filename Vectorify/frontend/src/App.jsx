import React from 'react'
import Navbar from '../components/Navbar'
import Title from '../components/Title'
import UploadBar from '../components/UploadBar'
import VerifyBar from '../components/VerifyBar'

function App() {
  return (
    <>

      <div className='flex flex-col items-center justify-center min-h-screen'>
      <Navbar />
        <Title />
        <UploadBar />
        <VerifyBar />
      </div>
    </>
  )
}

export default App
