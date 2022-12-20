'use strict';

chrome.tabs.query({'active': true, 'windowId': chrome.windows.WINDOW_ID_CURRENT},
   function(tabs){
      document.getElementById('page_url').setAttribute('value', tabs[0].url);
   }
);

// set the focus to the input box
document.getElementById("message").focus();

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('chatform').addEventListener('submit', function(evt) {
        evt.preventDefault();
        pushChat();
    });
})

function pushChat() {
    // if there is text to be sent...
    var wisdomText = document.getElementById('message');
    if (wisdomText && wisdomText.value && wisdomText.value.trim().length > 0) {
        // disable input to show we're sending it
        var wisdom = wisdomText.value.trim();
        //wisdomText.value = '...';
        wisdomText.locked = true;
        showRequest(wisdom);
        
        // recieve the message from server
        makeCorsRequest(document.getElementById('chatform'))
        
        // re-enable input
        wisdomText.value = '';
        wisdomText.locked = false;
    }
    // we always cancel form submission
    return false;
}
            
function showRequest(daText) {
        var conversationDiv = document.getElementById('conversation');
        var requestPara = document.createElement("p");
        requestPara.className = 'line request';
        requestPara.textContent = daText;
        conversationDiv.appendChild(requestPara);
        conversationDiv.scrollTop = conversationDiv.scrollHeight;
    }
 
 function showResponse(response) {
         var conversationDiv = document.getElementById('conversation');
         var responsePara = document.createElement("p");
         responsePara.className = 'line response';
         responsePara.textContent = response;
         conversationDiv.appendChild(responsePara);
         conversationDiv.scrollTop = conversationDiv.scrollHeight;
     }

function createCORSRequest(method, url) {
  var xhr = new XMLHttpRequest();
  if ("withCredentials" in xhr) {
    // XHR for Chrome/Firefox/Opera/Safari.
    xhr.open(method, url, true);
  } else if (typeof XDomainRequest != "undefined") {
    // XDomainRequest for IE.
    xhr = new XDomainRequest();
    xhr.open(method, url);
  } else {
    // CORS not supported.
    xhr = null;
  }
  return xhr;
}

// Make the actual CORS request.
function makeCorsRequest(form) {
//    var url = 'http://34.249.146.67:4242';
//    const data = new FormData(form);
//    var xhr = createCORSRequest('POST', url);
//    if (!xhr) {
//        return;
//    }
    
    // Response handlers.
    xhr.onload = function() {
    var text = xhr.responseText;
    showResponse(text)
    //alert('Response from CORS request to ' + url);
    };

    xhr.onerror = function() {
        alert('Woops, there was an error making the request.');
    };
    xhr.send(data);
}
