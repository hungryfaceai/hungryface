
// server/ws.js (Render) — minimal register/relay + mailbox
import http from 'http';
import express from 'express';
import { WebSocketServer } from 'ws';

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path:'/ws' });

const clients = new Map(); // fp -> ws
const mailbox = new Map(); // fp -> queued messages when offline

function deliver(toFp, msg){
  const ws = clients.get(toFp);
  if (ws && ws.readyState===ws.OPEN) ws.send(JSON.stringify(msg));
  else { const arr = mailbox.get(toFp) || []; arr.push(msg); mailbox.set(toFp, arr); }
}

wss.on('connection', (ws, req)=>{
  let myFp = null;
  ws.on('message', (buf)=>{
    let m; try { m = JSON.parse(buf.toString()); } catch { return; }
    if (m.op === 'register'){
      myFp = m.fp; clients.set(myFp, ws);
      const q = mailbox.get(myFp) || []; mailbox.delete(myFp);
      for (const mm of q) ws.send(JSON.stringify(mm));
      return;
    }
    if (m.op === 'relay' && m.to){ deliver(m.to, m); return; } // relays eph-hello/eph-reply/enc with their `sid`
    if (m.op === 'pair-init' || m.op === 'pair-ack'){ deliver(m.to, m); return; }
  });
  ws.on('close', ()=>{ if (myFp) clients.delete(myFp); });
});

server.listen(process.env.PORT || 3000, ()=> console.log('WS up'));
