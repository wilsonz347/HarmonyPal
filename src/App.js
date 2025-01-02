import React, { useState, useRef, useEffect} from 'react';
import { Send, Bot } from 'lucide-react';

const App = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (input.trim()) {
      setMessages([...messages, { text: input, sender: 'user' }]);
      setInput('');

      try {
        const response = await fetch('http://localhost:5000/generate_response', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ question: input }),
        });

        const data = await response.json();

        setMessages(prevMessages => [...prevMessages, {
          text: data.response,
          sender: 'bot'
        }]);
      } catch (error) {
        console.error('Error:', error);
        setMessages(prevMessages => [...prevMessages, {
          text: "Sorry, I'm having trouble connecting right now.",
          sender: 'bot'
        }]);
      }
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-100 to-white p-4">
      <div className="max-w-2xl mx-auto">
        {/* Animated Robot */}
        <div className="flex justify-center mb-2 animate-bounce pt-4">
          <div className="relative">
            <Bot size={48} className="text-blue-500"/>
            <div className="absolute -right-2 -top-2 w-3 h-3 bg-green-400 rounded-full animate-pulse"/>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="bg-white rounded-lg shadow-lg p-4 mb-4 h-96 overflow-y-auto">
          {messages.map((msg, idx) => (
              <div key={idx} className={`mb-4 ${msg.sender === 'user' ? 'text-right' : ''}`}>
                <div className={`inline-block p-3 rounded-lg ${
                    msg.sender === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'
                }`}>
                  {msg.text}
                </div>
              </div>
          ))}
          {messages.length === 0 && (
              <div className="text-center text-gray-500 mt-32">
                How are you feeling today? I'm here to listen.
              </div>
          )}
          <div ref={messagesEndRef}/>
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="flex gap-2">
          {/* Clear Chat Button */}
          <button
              type="button"
              onClick={clearChat}
              className="bg-red-500 text-white p-2 rounded-lg hover:bg-red-600 transition-colors">
            Clear Chat
          </button>

          <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 p-3 rounded-lg border focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
              type="submit"
              className="bg-blue-500 text-white p-3 rounded-lg hover:bg-blue-600 transition-colors"
          >
            <Send size={20}/>
          </button>
        </form>

      </div>
    </div>
  );
};

export default App;