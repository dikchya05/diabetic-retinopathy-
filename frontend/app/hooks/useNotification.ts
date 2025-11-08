import { useState, useEffect } from 'react';

export type NotificationType = 'success' | 'error' | 'info' | 'warning';

export interface Notification {
  message: string;
  type: NotificationType;
}

export function useNotification() {
  const [notification, setNotification] = useState<Notification | null>(null);

  // Auto-hide notification after 5 seconds
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => {
        setNotification(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  const showNotification = (message: string, type: NotificationType = 'info') => {
    setNotification({ message, type });
  };

  const showSuccess = (message: string) => {
    setNotification({ message, type: 'success' });
  };

  const showError = (message: string) => {
    setNotification({ message, type: 'error' });
  };

  const showInfo = (message: string) => {
    setNotification({ message, type: 'info' });
  };

  const showWarning = (message: string) => {
    setNotification({ message, type: 'warning' });
  };

  const hideNotification = () => {
    setNotification(null);
  };

  return {
    notification,
    showNotification,
    showSuccess,
    showError,
    showInfo,
    showWarning,
    hideNotification,
  };
}
