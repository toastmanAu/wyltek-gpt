// Minimal service worker — exists so the app is "installable" and can act
// as a Web Share Target. Browsers require a registered SW with at least
// one fetch handler (even a no-op one) before they'll surface the install
// prompt or accept share-target POSTs.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", () => {
  // Pass-through. Add caching strategies here later if you want offline
  // converters / cached UI shell.
});
