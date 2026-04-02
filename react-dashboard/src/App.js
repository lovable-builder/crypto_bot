import logo from './logo.svg';
import './App.css';
import { useState, useEffect } from 'react';

function App() {
  const [isLightTheme, setIsLightTheme] = useState(() => {
    const savedTheme = localStorage.getItem('theme');
    return savedTheme === 'light';
  });

  useEffect(() => {
    if (isLightTheme) {
      document.body.classList.add('light-theme');
    } else {
      document.body.classList.remove('light-theme');
    }
    localStorage.setItem('theme', isLightTheme ? 'light' : 'dark');
  }, [isLightTheme]);

  const toggleTheme = () => {
    setIsLightTheme(!isLightTheme);
  };

  return (
    <div className="App">
      <header className="App-header">
        <button 
          onClick={toggleTheme}
          style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            padding: '8px 16px',
            borderRadius: '6px',
            border: '1px solid var(--border)',
            background: 'var(--surface)',
            color: 'var(--muted)',
            cursor: 'pointer',
            fontSize: '14px',
            fontWeight: '600',
            transition: 'all 0.2s'
          }}
          title={isLightTheme ? 'Switch to dark theme' : 'Switch to light theme'}
        >
          {isLightTheme ? '🌙' : '☀️'}
        </button>
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
