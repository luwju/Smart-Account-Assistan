import ChatWidget from '../components/ChatWidget/ChatWidget';

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Your main website content */}
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-center text-gray-800 mb-4">
          Welcome to Meboat
        </h1>
        <p className="text-center text-gray-600 mb-8">
          Your trusted banking partner
        </p>
        
        {/* Sample website content */}
        <div className="max-w-2xl mx-auto text-center">
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Me Made Simple
            </h2>
            <p className="text-gray-600 mb-4">
              Open accounts, apply for loans, and manage your finances with ease.
            </p>
            <button className="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700 transition-colors">
              welcome to u
            </button>
          </div>
          
          <p className="text-gray-500 text-sm">
            Need help? Click the chat icon in the bottom right corner!
          </p>
        </div>
      </main>

      {/* Floating Chat Widget */}
      <ChatWidget />
    </div>
  );
}