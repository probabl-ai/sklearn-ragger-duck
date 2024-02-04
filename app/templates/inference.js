const converter = new showdown.Converter();

let ws;
let response = "";
let conversationStarted = false;
let dontScroll = false;

function setButton(value, disabled) {
    var button = document.getElementById('send');
    button.innerHTML = value;
    button.disabled = disabled
    if(disabled) {
        button.classList.add("blink_me");
    } else {
        button.classList.remove("blink_me");
    }
}

function pickModel(model){
    messageText.value = "!model " + model;
    messageText.focus();
    sendQuery();
    messageText.value = "";
    messageText.focus();
    return false;
}

function setPromptTemplate(template, cursorLocation){
    messageText.value = template;
    messageText.focus();
    messageText.selectionEnd = cursorLocation;
}

function localCommandExecuted(e){
    const messageText = document.getElementById("messageText");
    switch(messageText.value.toLowerCase()){
        {% for template in conf.PROMPT_TEMPLATES %}
        case '#{{ template[0] }}':
            setPromptTemplate("{{ template[1]|safe }}", {{ template[2] }});
            if (e) e.preventDefault();
            return true;
        {% endfor %}
        default:
            return false;
    }
}

function sendQuery() {
    if (localCommandExecuted()) return;
    if (!ws) return;
    conversationStarted = true;
    var message = document.getElementById('messageText').value;
    var temp = document.getElementById('tempValue').innerHTML;
    var cutoff = document.getElementById('cutoffValue').innerHTML;
    var min_top_k = document.getElementById('minTopKValue').innerHTML;
    var max_top_k = document.getElementById('maxTopKValue').innerHTML;

    if (message === "") {
        return;
    }
    payload = {
        query: message,
        temperature: parseFloat(temp),
        cutoff: parseFloat(cutoff),
        min_top_k: parseInt(min_top_k),
        max_top_k: parseInt(max_top_k),
    }
    ws.send(JSON.stringify(payload));
    setButton("{{ res.BUTTON_PROCESSING }}", true)
}

function submitForm(event) {
    event.preventDefault();
    sendQuery();
}

function appendButtons() {
    var messages = document.getElementById('messages');
    var div = messages;

    var span = document.createElement('span');
    span.innerHTML="<span class='btn btn-primary clip-button' onclick='copyClipboard(this)'><img src='/static/img/copy.svg'>&nbsp;&nbsp;<span id='clip-button-label'>Copy to clipboard</span></span>"
    div.appendChild(span);
}

function copyClipboard(e) {
    var messages = document.getElementById('messages');
    var div = messages.firstChild.firstChild;
    content = div.innerText;
    navigator.clipboard.writeText(content);
    label = document.getElementById("clip-button-label")
    label.innerHTML = "Copied!"
    setTimeout(()=>{
        label.innerText = "Copy to clipboard";
    },
    2000)
}

function updateResponseTokens() {
    responseTokenCountValue = document.getElementById("responseTokenCountValue");
    encoded = llamaTokenizer.encode(response)
    responseTokenCountValue.innerText=encoded.length;
    responseTokenCount.classList.remove("d-none");
}

function hideResponseTokens() {
    responseTokenCount = document.getElementById("responseTokenCount");
    responseTokenCount.classList.add("d-none");
}

function connect() {
    let wsBaseUrl = "{{ wsurl }}";
    if (wsBaseUrl === "") {
        let wsProtocol = "https:" === document.location.protocol ? 'wss://' : 'ws://'
        wsBaseUrl = wsProtocol + window.location.host;
    }
    ws = new WebSocket(wsBaseUrl + "/inference");
    ws.onmessage = function (event) {
        var messages = document.getElementById('messages');
        var data = JSON.parse(event.data);
        handleResBotResponse(data ,messages)

        // Scroll to the bottom of the chat (don't auto scroll if user has scrolled manually)
        if (!dontScroll){
            messages.scrollTop = Math.floor(messages.scrollHeight - messages.offsetHeight)
        }
    };
    ws.onopen = function() {
        setButton("{{ res.BUTTON_SEND }}", false);
    };

    ws.onclose = function(e) {
        setButton("{{ res.BUTTON_WAIT }}", true);
        console.log('Socket is closed. Reconnect will be attempted in 1 second.', e.reason);
        setTimeout(function() {
            connect();
        }, 1000);
    };
}

function handleResBotResponse(data, messages) {
    switch(data.type) {
        case "start":
            dontScroll = false;
            messages.innerHTML = '';
            response = "";
            var div = document.createElement('div');
            div.className = 'server-message';
            var p = document.createElement('p');
            response += data.message;
            p.innerHTML = converter.makeHtml(response);
            div.appendChild(p);
            messages.appendChild(div);
            updateResponseTokens();
            break;

        case "stream":
            setButton("{{ res.BUTTON_TYPING }}", true);
            var p = messages.lastChild.lastChild;
            response += data.message;
            p.innerHTML = converter.makeHtml(response);
            updateResponseTokens();
            break;

        case "end":
            var p = messages.lastChild.lastChild;
            p.innerHTML = converter.makeHtml(response);
            setButton("{{ res.BUTTON_SEND }}", false);
            updateResponseTokens();
            appendButtons();
            break;

        case "done":
            hideResponseTokens();
            setButton("{{ res.BUTTON_SEND }}", false);
            break;

        case "info":
            messages.innerHTML = '';
            var div = document.createElement('div');
            div.className = 'server-message';
            var p = document.createElement('p');
            // p.innerHTML = converter.makeHtml(data.message);
            p.innerHTML = data.message;
            div.appendChild(p);
            messages.appendChild(div);
            hideResponseTokens();
            break;

        case "system":
            messages.innerHTML = '';
            var div = document.createElement('div');
            div.className = 'server-message';
            var p = document.createElement('p');
            p.innerHTML = data.message;
            div.appendChild(p);
            messages.appendChild(div);
            hideResponseTokens();
            setButton("{{ res.BUTTON_SEND }}", false);
            break;

        case "error":
            messages.innerHTML = '';
            var div = document.createElement('div');
            div.className = 'server-message';
            var p = document.createElement('p');
            // p.innerHTML = converter.makeHtml(data.message);
            p.innerHTML = data.message;
            div.appendChild(p);
            messages.appendChild(div);
            hideResponseTokens();
            setButton("{{ res.BUTTON_SEND }}", false);
            break;
    }
}

document.addEventListener("DOMContentLoaded", function(event) {
    let messages=document.getElementById("messages");
    let topbox=document.getElementById("top-box");

    ['touchmove','mousedown','select','wheel'].forEach((evt) => {
        messages.addEventListener(evt, (e) => {
            if(Math.floor(messages.scrollTop) === Math.floor(messages.scrollHeight - messages.offsetHeight)) {
                dontScroll = false;
            }
            else{
                dontScroll = true;
            }
        });
    });


    const temperature_slider = document.getElementById("tempSlider");
    const temperature_label = document.getElementById("tempValue");
    temperature_slider.addEventListener("input", () => {
        temperature_label.innerText=temperature_slider.value;
    });
    const cutoff_slider = document.getElementById("cutoffSlider");
    const cutoff_label = document.getElementById("cutoffValue");
    cutoff_slider.addEventListener("input", () => {
        cutoff_label.innerText=cutoff_slider.value;
    });
    const min_top_k_slider = document.getElementById("minTopKSlider");
    const min_top_k_label = document.getElementById("minTopKValue");
    min_top_k_slider.addEventListener("input", () => {
        min_top_k_label.innerText=min_top_k_slider.value;
    });
    const max_top_k_slider = document.getElementById("maxTopKSlider");
    const max_top_k_label = document.getElementById("maxTopKValue");
    max_top_k_slider.addEventListener("input", () => {
        max_top_k_label.innerText=max_top_k_slider.value;
    });

    const button = document.getElementById('send');
    const tokenCount = document.getElementById("tokenCount");
    const tokenCountValue = document.getElementById("tokenCountValue");
    const messageText = document.getElementById("messageText");
    messageText.addEventListener("keydown", (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            if (!button.disabled) sendQuery();
            e.preventDefault();
        }
        if (e.key === "Tab") {
            localCommandExecuted(e);
            e.preventDefault();
        }
    })
    messageText.focus();

    messageText.addEventListener("input", () => {
        encoded = llamaTokenizer.encode(messageText.value)
        if (encoded.length > 0) {
            tokenCount.classList.remove("d-none");
            tokenCountValue.innerText=encoded.length
        } else {
            tokenCount.classList.add("d-none");
        }
        if (!button.disabled) {
            if (encoded.length > {{ conf.CONTEXT_TOKENS }}) {
                button.disabled = true;
                tokenCount.classList.add("tokenLimitExceeded");
            } else {
                button.disabled = false;
                tokenCount.classList.remove("tokenLimitExceeded");
            }
        }
    });

    connect();
});
