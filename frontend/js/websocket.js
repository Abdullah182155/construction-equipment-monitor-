/**
 * WebSocket Client Module
 * Handles real-time connection to the FastAPI backend with auto-reconnect.
 */

class WSClient {
    constructor(url) {
        this.url = url || `ws://${window.location.host}/ws`;
        this.ws = null;
        this.reconnectDelay = 1000;
        this.maxReconnectDelay = 30000;
        this.isConnected = false;
        this.intentionalClose = false;

        // Event callbacks
        this.onFrame = null;       // (base64ImageData) => {}
        this.onEvent = null;       // (equipmentEvent) => {}
        this.onSummary = null;     // (summaryData) => {}
        this.onStatus = null;      // (statusData) => {}
        this.onConnect = null;     // () => {}
        this.onDisconnect = null;  // () => {}
    }

    connect() {
        if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        this.intentionalClose = false;

        try {
            this.ws = new WebSocket(this.url);
        } catch (e) {
            console.error('WebSocket creation failed:', e);
            this._scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            console.log('[WS] Connected to', this.url);
            this.isConnected = true;
            this.reconnectDelay = 1000;
            if (this.onConnect) this.onConnect();
        };

        this.ws.onmessage = (event) => {
            this._handleMessage(event.data);
        };

        this.ws.onclose = (event) => {
            console.log('[WS] Connection closed:', event.code, event.reason);
            this.isConnected = false;
            if (this.onDisconnect) this.onDisconnect();

            if (!this.intentionalClose) {
                this._scheduleReconnect();
            }
        };

        this.ws.onerror = (error) => {
            console.error('[WS] Error:', error);
        };
    }

    disconnect() {
        this.intentionalClose = true;
        if (this.ws) {
            this.ws.close();
        }
    }

    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(typeof data === 'string' ? data : JSON.stringify(data));
        }
    }

    sendCommand(command) {
        this.send({ command });
    }

    _handleMessage(rawData) {
        try {
            const msg = JSON.parse(rawData);

            switch (msg.type) {
                case 'frame':
                    if (this.onFrame) this.onFrame(msg.data);
                    break;
                case 'event':
                    if (this.onEvent) this.onEvent(msg.data);
                    break;
                case 'summary':
                    if (this.onSummary) this.onSummary(msg.data);
                    break;
                case 'status':
                    if (this.onStatus) this.onStatus(msg.data);
                    break;
                default:
                    console.warn('[WS] Unknown message type:', msg.type);
            }
        } catch (e) {
            console.warn('[WS] Failed to parse message:', e);
        }
    }

    _scheduleReconnect() {
        console.log(`[WS] Reconnecting in ${this.reconnectDelay}ms...`);
        setTimeout(() => {
            this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, this.maxReconnectDelay);
            this.connect();
        }, this.reconnectDelay);
    }
}

// Export as global
window.WSClient = WSClient;
