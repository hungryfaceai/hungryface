import express from 'express';
import { WebSocketServer } from 'ws';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();

// Serve static files (HTML, JS, CSS) from repo
app.use(express.static(__dirname));

const server = app.listen(process.env.PORT || 3000, () => {
  console.log(`Server running on port ${server.address().port}`);
});

// Simple WebSocket signaling
const wss = new WebSocketServer({ server });
const rooms = {};

wss.on('connection', ws => {
  ws.on('message', message => {
    const data = JSON.parse(message);
    if (data.type === 'join') {
      if (!rooms[data.room]) rooms[data.room] = [];
      rooms[data.room].push(ws);
      ws.room = data.room;
    } else {
      // Broadcast to others in same room
      rooms[ws.room]?.forEach(client => {
        if (client !== ws && client.readyState === 1) {
          client.send(JSON.stringify(data));
        }
      });
    }
  });

  ws.on('close', () => {
    if (ws.room) {
      rooms[ws.room] = rooms[ws.room]?.filter(c => c !== ws);
    }
  });
});
