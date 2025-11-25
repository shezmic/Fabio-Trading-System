class WebSocketClient {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.handlers = {};
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
            console.log('Connected to WebSocket');
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            const type = message.type;

            if (this.handlers[type]) {
                this.handlers[type].forEach(handler => handler(message.payload));
            }
        };

        this.ws.onclose = () => {
            console.log('Disconnected');
            // Reconnect logic?
        };
    }

    on(type, handler) {
        if (!this.handlers[type]) {
            this.handlers[type] = [];
        }
        this.handlers[type].push(handler);
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

export default WebSocketClient;
