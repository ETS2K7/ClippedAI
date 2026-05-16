
const DB_NAME = "ClippedAI_Storage";
const STORE_NAME = "PendingUploads";

export async function storePendingFile(file: File): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onupgradeneeded = (event: any) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };

    request.onsuccess = (event: any) => {
      const db = event.target.result;
      const transaction = db.transaction(STORE_NAME, "readwrite");
      const store = transaction.objectStore(STORE_NAME);
      
      // Clear previous and store new
      store.clear();
      const putRequest = store.put(file, "current_file");
      
      putRequest.onsuccess = () => resolve();
      putRequest.onerror = () => reject(new Error(putRequest.error?.message || "Failed to store file"));
    };

    request.onerror = () => reject(new Error(request.error?.message || "Failed to open indexedDB"));
  });
}

export async function getPendingFile(): Promise<File | null> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);

    request.onupgradeneeded = (event: any) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };

    request.onsuccess = (event: any) => {
      const db = event.target.result;
      const transaction = db.transaction(STORE_NAME, "readonly");
      const store = transaction.objectStore(STORE_NAME);
      const getRequest = store.get("current_file");

      getRequest.onsuccess = () => resolve(getRequest.result || null);
      getRequest.onerror = () => reject(new Error(getRequest.error?.message || "Failed to get file"));
    };

    request.onerror = () => reject(new Error(request.error?.message || "Failed to open indexedDB"));
  });
}

export async function clearPendingFile(): Promise<void> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onsuccess = (event: any) => {
      const db = event.target.result;
      const transaction = db.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).clear();
      resolve();
    };
    request.onerror = () => reject(new Error(request.error?.message || "Failed to open indexedDB"));
  });
}
