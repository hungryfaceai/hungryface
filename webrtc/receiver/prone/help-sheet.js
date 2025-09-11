// Wire up the "Close" button inside the loaded help sheet
(function(){
  const overlay = document.getElementById('helpOverlay');
  function close(){ overlay?.classList.remove('open'); overlay?.setAttribute('aria-hidden','true'); }
  // delegate since content is injected
  document.addEventListener('click', (e) => {
    const t = e.target;
    if (t && t.id === 'helpClose') {
      e.preventDefault();
      e.stopPropagation();
      close();
    }
  }, true);
})();
