import React from 'react'
import '../App.css';

const Header = () => {

  const [isOpen, setIsOpen] = useState(false);


  return (
    <nav className='bg-white border-b border-gray-100'>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"></div>
    </nav>
  )
}

export default Header