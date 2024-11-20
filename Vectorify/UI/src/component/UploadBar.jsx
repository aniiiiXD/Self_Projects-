import React, { useState } from 'react';
import { Upload, CheckCircle, AlertCircle, User, Lock, UserCheck } from 'lucide-react';

const UploadBar = () => {
  const [file, setFile] = useState(null);
  const [verifier, setVerifier] = useState('');
  const [message, setMessage] = useState('');
  const [status, setStatus] = useState('idle'); // idle, loading, success, error
  const [searchTerm, setSearchTerm] = useState(''); // New state for search input

  // Dummy verifier data - in real app, this would come from your backend
  const availableVerifiers = [
    { id: 1, name: 'MAdhusudhan', role: 'Senior Developer' },
    { id: 2, name: 'SR Ghorpade', role: 'Tech Lead' },
    { id: 3, name: 'Rekha', role: 'Project Manager' },
    { id: 4, name: 'MAyukh', role: 'Senior Developer' },
    { id: 5, name: 'Bata', role: 'Team Lead' }
  ];

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setStatus('idle');
    setMessage('');
  };

  const handleVerifierChange = (event) => {
    setVerifier(event.target.value);
    setStatus('idle');
    setMessage('');
  };

  const handleFileUpload = async (event) => {
    event.preventDefault();
    
    if (!file) {
      setStatus('error');
      setMessage('Please select a file to upload.');
      return;
    }

    if (!verifier) {
      setStatus('error');
      setMessage('Please select a verifier for the file.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('verifier', verifier);
    
    try {
      setStatus('loading');
      const response = await fetch('http://localhost:3002/upload-files', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Upload failed');
      const data = await response.json();
      setStatus('success');
      setMessage(`File uploaded successfully and assigned to ${verifier} for verification!`);
    } catch (error) {
      console.error('Error uploading file:', error);
      setStatus('error');
      setMessage('Error uploading file');
    }
  };

  // Filter verifiers based on search term
  const filteredVerifiers = availableVerifiers.filter(v =>
    v.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="min-h-screen flex items-center justify-center bg-black w-full">
      <div className="px-4 py-12 max-w-4xl w-full space-y-8">
        {/* File upload section */}
        <div className="space-y-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center">
              <Upload className="w-5 h-5 text-white/80" />
            </div>
            <h2 className="text-2xl font-light text-white/90">Upload File for Verification</h2>
          </div>

          <form onSubmit={handleFileUpload} className="space-y-6">
            <div className="relative group">
              <input
                type="file"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="flex flex-col items-center justify-center w-full h-48 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 transition-all cursor-pointer backdrop-blur-sm"
              >
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <Upload className="w-10 h-10 mb-3 text-white/40 group-hover:text-white/60" />
                  <p className="mb-2 text-xl text-white/80 group-hover:text-white/90">
                    {file ? file.name : 'Drop your file here'}
                  </p>
                  <p className="text-sm text-white/40 group-hover:text-white/60">
                    Click to upload or drag and drop
                  </p>
                </div>
              </label>
            </div>

            {/* Verifier selection */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-gray-700 to-gray-800 flex items-center justify-center">
                  <UserCheck className="w-4 h-4 text-white/80" />
                </div>
                <h3 className="text-lg font-light text-white/90">Select Verifier</h3>
              </div>

              {/* Search bar */}
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search verifier by name"
                className="w-full p-3 rounded-lg border border-white/10 bg-white/5 text-white/80 placeholder-white/60"
              />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {filteredVerifiers.map((v) => (
                  <label
                    key={v.id}
                    className={`relative flex items-center p-4 rounded-lg cursor-pointer backdrop-blur-sm transition-all
                      ${verifier === v.name 
                        ? 'bg-white/10 border-2 border-blue-500/50'
                        : 'bg-white/5 border border-white/10 hover:bg-white/10'}`}
                  >
                    <input
                      type="radio"
                      name="verifier"
                      value={v.name}
                      checked={verifier === v.name}
                      onChange={handleVerifierChange}
                      className="hidden"
                    />
                    <div className="flex-1">
                      <h4 className="text-white/90 font-medium">{v.name}</h4>
                      <p className="text-sm text-white/60">{v.role}</p>
                    </div>
                    {verifier === v.name && (
                      <CheckCircle className="w-5 h-5 text-blue-500 absolute right-4" />
                    )}
                  </label>
                ))}
              </div>
            </div>

            {file && (
              <button
                type="submit"
                disabled={status === 'loading' || !verifier}
                className={`w-full py-3 px-4 rounded-lg text-center text-white/90 font-light transition-all flex items-center justify-center space-x-2
                  ${status === 'loading' || !verifier
                    ? 'bg-white/10 cursor-not-allowed'
                    : 'bg-white/5 hover:bg-white/10 backdrop-blur-sm'}`}
              >
                <Upload className="w-5 h-5" />
                <span>{status === 'loading' ? 'Uploading...' : 'Upload for Verification'}</span>
              </button>
            )}
          </form>
        </div>

        {/* Status message */}
        {message && (
          <div className={`flex items-center justify-center space-x-2 p-4 rounded-lg backdrop-blur-sm
            ${status === 'success' ? 'bg-green-500/5 text-green-300' : ''}
            ${status === 'error' ? 'bg-red-500/5 text-red-300' : ''}`}
          >
            {status === 'success' && <CheckCircle className="w-5 h-5" />}
            {status === 'error' && <AlertCircle className="w-5 h-5" />}
            <span className="font-light">{message}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadBar;
