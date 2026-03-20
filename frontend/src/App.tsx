import React from 'react'
import { Layout } from './components/Layout'
import { ImageViewer } from './components/ImageViewer'
import { SpectralChart } from './components/SpectralChart'
import { MatchTable } from './components/MatchTable'
import './App.css'

const App: React.FC = () => {
  return (
    <Layout>
      <div className="main-grid">
        <div className="main-col left-col">
          <ImageViewer />
          <SpectralChart />
        </div>
        <div className="main-col right-col">
          <MatchTable />
        </div>
      </div>
    </Layout>
  )
}

export default App
