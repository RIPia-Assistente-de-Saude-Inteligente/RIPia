function sendMessage() {
    const input = document.getElementById('user-input');
    const message = input.value.trim();
    if (!message) return;
    const chatBox = document.getElementById('chat-box');
    chatBox.innerHTML += `<div class="user-msg"><b>You:</b> ${message}</div>`;
    input.value = '';
    showLoading(true);

    fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: message})
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        chatBox.innerHTML += `<div class="ia-msg"><b>AI:</b> ${data.response}</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    })
    .catch(() => {
        showLoading(false);
        chatBox.innerHTML += `<div class="ia-msg"><b>AI:</b> An error occurred. Please try again.</div>`;
    });
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}
