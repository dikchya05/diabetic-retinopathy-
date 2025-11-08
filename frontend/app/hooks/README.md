# useNotification Hook

A reusable custom hook for displaying toast notifications in your React components.

## Usage

### Basic Example

```tsx
import { useNotification } from './hooks/useNotification';
import Toast from './components/Toast';

function MyComponent() {
  const { notification, showSuccess, showError, hideNotification } = useNotification();

  const handleAction = async () => {
    try {
      // Your async operation
      await saveData();
      showSuccess('Data saved successfully!');
    } catch (error) {
      showError('Failed to save data. Please try again.');
    }
  };

  return (
    <div>
      {/* Your component content */}
      <button onClick={handleAction}>Save</button>

      {/* Toast notification */}
      {notification && (
        <Toast notification={notification} onClose={hideNotification} />
      )}
    </div>
  );
}
```

### API

#### Hook Methods

- **`showSuccess(message: string)`** - Show a success notification (green)
- **`showError(message: string)`** - Show an error notification (red)
- **`showInfo(message: string)`** - Show an info notification (blue)
- **`showWarning(message: string)`** - Show a warning notification (yellow/orange)
- **`showNotification(message: string, type: NotificationType)`** - Show a notification with custom type
- **`hideNotification()`** - Manually hide the current notification

#### Hook Return Values

- **`notification`** - Current notification object (or null)

### Notification Types

- `'success'` - Green border, checkmark icon
- `'error'` - Red border, error icon
- `'info'` - Blue border, info icon
- `'warning'` - Orange border, warning icon

### Features

- ✨ Auto-dismisses after 5 seconds
- ❌ Manual close button
- 🎨 Four notification types (success, error, info, warning)
- 📱 Mobile responsive
- 🎭 Smooth slide-in animation
- ♻️ Fully reusable across components

### Examples

```tsx
// Success notification
showSuccess('Patient created successfully!');

// Error notification
showError('Failed to load data');

// Info notification
showInfo('New scan results available');

// Warning notification
showWarning('Your session will expire in 5 minutes');

// Custom notification
showNotification('Custom message', 'info');
```
