import React from 'react'
import { Sidebar } from './Sidebar'
import { StatusBar } from './StatusBar'
import './Layout.css'

export const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="app-shell">
      <div className="layout-container">
        <Sidebar />
        <div className="main-content">
          <div className="content-area">
            {children}
          </div>
        </div>
      </div>
      <StatusBar />
    </div>
  )
}
