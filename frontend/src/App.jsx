import { useState } from 'react'
import Recommendations from './components/Recommendations'
import Search from './components/Search'
import ColdStart from './components/ColdStart'

function App() {
    const [activeTab, setActiveTab] = useState('recommendations')
    const [userId, setUserId] = useState('')
    const [inputUserId, setInputUserId] = useState('')

    const handleSetUser = () => {
        if (inputUserId.trim()) {
            setUserId(inputUserId.trim())
        }
    }

    return (
        <div className="container">
            <div className="header">
                <h1>LLM-Enhanced Recommendation System</h1>
                <p>Production-ready recommendation engine with 540K+ transactions</p>

                <div className="input-group" style={{ maxWidth: '400px', marginTop: '15px' }}>
                    <label>User ID:</label>
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <input
                            type="text"
                            value={inputUserId}
                            onChange={(e) => setInputUserId(e.target.value)}
                            placeholder="Enter user ID (e.g., 12346)"
                            onKeyPress={(e) => e.key === 'Enter' && handleSetUser()}
                        />
                        <button className="button" onClick={handleSetUser}>
                            Set User
                        </button>
                    </div>
                    {userId && <p style={{ marginTop: '5px', fontSize: '14px', color: '#666' }}>
                        Current User: <strong>{userId}</strong>
                    </p>}
                </div>

                <div className="nav">
                    <button
                        className={activeTab === 'recommendations' ? 'active' : ''}
                        onClick={() => setActiveTab('recommendations')}
                    >
                        Recommendations
                    </button>
                    <button
                        className={activeTab === 'search' ? 'active' : ''}
                        onClick={() => setActiveTab('search')}
                    >
                        Natural Language Search
                    </button>
                    <button
                        className={activeTab === 'coldstart' ? 'active' : ''}
                        onClick={() => setActiveTab('coldstart')}
                    >
                        Cold Start (New Users)
                    </button>
                </div>
            </div>

            {activeTab === 'recommendations' && <Recommendations userId={userId} />}
            {activeTab === 'search' && <Search userId={userId} />}
            {activeTab === 'coldstart' && <ColdStart />}
        </div>
    )
}

export default App
