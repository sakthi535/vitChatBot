// generate_startMessage();
// A wild one appeared
var bot = true

$(function() {
  var INDEX = 0; 

  if(bot){
    generate_startMessage();
    bot = false;
  }

  $("#chat-listen").click(function(e){
    e.preventDefault();
    var recognition = new webkitSpeechRecognition();

    $("#chat-listen").css("background-color","red");
    $("#chat-listen").css("color","white");

    var temp = document.getElementById('chat-input').value;
    document.getElementById('chat-input').value = "Text is recording ..."


    recognition.lang = document.querySelector('input[name="chat-lang"]:checked').value;
    console.log(recognition.lang);
    recognition.onresult = function(event) {

        console.log(event);
        document.getElementById('chat-input').value = temp + event.results[0][0].transcript;
        $("#chat-listen").css("background-color","white");
        $("#chat-listen").css("color","blue");
        sendText(e);


    }
    recognition.start();


  })


  async function postData(query) {
    // Default options are marked with *
    url = 'http://127.0.0.1:5000'
    lang = document.querySelector('input[name="chat-lang"]:checked').value
    // let query = document.getElementById()
    data = {"query" : query, "lang" : lang}
    const response = await fetch(url, {
      method: 'POST', // *GET, POST, PUT, DELETE, etc.
      mode: 'cors', // no-cors, *cors, same-origin
      cache: 'no-cache', // *default, no-cache, reload, force-cache, only-if-cached
      credentials: 'same-origin', // include, *same-origin, omit
      headers: {
        'Content-Type': 'application/json'
        // 'Content-Type': 'application/x-www-form-urlencoded',
      },
      redirect: 'follow', // manual, *follow, error
      referrerPolicy: 'no-referrer', // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
      body: JSON.stringify(data) // body data type must match "Content-Type" header
    });
    return response.json(); // parses JSON response into native JavaScript objects
  }

  

  $("#text-input-form").submit(function(e) {
    e.preventDefault();
    sendText(e);
});


  function sendText(e) {
    e.preventDefault();

    var msg = $("#chat-input").val(); 
    if(msg.trim() == ''){
      return false;
    }
    generate_message(msg, 'self');
    var buttons = [
        {
          name: 'Existing User',
          value: 'existing'
        },
        {
          name: 'New User',
          value: 'new'
        }
      ];
    setTimeout(function() {      

      var cheatRes;
      var botReply = postData(msg)
      .then((data) => {
        console.log(data['data']);
        generate_message(data['data'], 'user');  

        cheatRes = data['data']; // JSON data parsed by `data.json()` call
      });
      console.log(msg)
      console.log(cheatRes)
      // generate_message(botReply, 'user');  
    }, 1000)
    
  }

  function generate_startMessage(){
    msg = "Please Select language Preference"

    type = "user"
    INDEX++;
    var str="";
    str += "<div id='cm-msg-"+INDEX+"' class=\"chat-msg "+type+"\">";
    str += "          <div class=\"msg-avatar\">";
    str += "            <i class=\"material-icons\" >account_circle</i>";
    str += "          <\/div>";
    str += "          <div class=\"cm-msg-text\">";
    str += msg;
    
    str += " <br><input type='radio' name = 'chat-lang' value = 'en-IN' checked = 'checked'>";
    str += " <label> English </label>";

    str += " <br><input type='radio' name = 'chat-lang' value = 'hi-IN'>"; 
    str += " <label> Hindi </label>";
    
    str += " <br><input type='radio' name = 'chat-lang' value = 'ta-IN'>"; 
    str += " <label> Tamil </label>";

    str += "          <\/div>";
    str += "        <\/div>";
    $(".chat-logs").append(str);
    $("#cm-msg-"+INDEX).hide().fadeIn(300);
    if(type == 'self'){
     $("#chat-input").val(''); 
    }    
    $(".chat-logs").stop().animate({ scrollTop: $(".chat-logs")[0].scrollHeight}, 1000);    

  }
  
  function generate_message(msg, type) {
    INDEX++;
    var str="";

    


    str += "<div id='cm-msg-"+INDEX+"' class=\"chat-msg "+type+"\">";
    str += "          <div class=\"msg-avatar\">";
    str += "            <i class=\"material-icons\" >account_circle</i>";
    str += "          <\/div>";
    str += "          <div class=\"cm-msg-text\">";
    str += msg;
    str += "          <\/div>";
    str += "        <\/div>";
    $(".chat-logs").append(str);
    $("#cm-msg-"+INDEX).hide().fadeIn(300);
    
    
    
    
    if(type == 'self'){
     $("#chat-input").val(''); 
    }    
    $(".chat-logs").stop().animate({ scrollTop: $(".chat-logs")[0].scrollHeight}, 1000);    
  }  
  
  function generate_button_message(msg, buttons){    
    /* Buttons should be object array 
      [
        {
          name: 'Existing User',
          value: 'existing'
        },
        {
          name: 'New User',
          value: 'new'
        }
      ]
    */
    INDEX++;
    var btn_obj = buttons.map(function(button) {
       return  "              <li class=\"button\"><a href=\"javascript:;\" class=\"btn btn-primary chat-btn\" chat-value=\""+button.value+"\">"+button.name+"<\/a><\/li>";
    }).join('');
    var str="";
    str += "<div id='cm-msg-"+INDEX+"' class=\"chat-msg user\">";
    str += "          <span class=\"msg-avatar\">";
    str += "            <img src=\"https:\/\/image.crisp.im\/avatar\/operator\/196af8cc-f6ad-4ef7-afd1-c45d5231387c\/240\/?1483361727745\">";
    str += "          <\/span>";
    str += "          <div class=\"cm-msg-text\">";
    str += msg;
    str += "          <\/div>";
    str += "          <div class=\"cm-msg-button\">";
    str += "            <ul>";   
    str += btn_obj;
    str += "            <\/ul>";
    str += "          <\/div>";
    str += "        <\/div>";
    $(".chat-logs").append(str);
    $("#cm-msg-"+INDEX).hide().fadeIn(300);   
    $(".chat-logs").stop().animate({ scrollTop: $(".chat-logs")[0].scrollHeight}, 1000);
    $("#chat-input").attr("disabled", true);
  }
  
  $(document).delegate(".chat-btn", "click", function() {
    var value = $(this).attr("chat-value");
    var name = $(this).html();
    $("#chat-input").attr("disabled", false);
    generate_message(name, 'self');
  })
  
  $("#chat-circle").click(function() {    
    $("#chat-circle").toggle('scale');
    $(".chat-box").toggle('scale');
  })
  
  $(".chat-box-toggle").click(function() {
    $("#chat-circle").toggle('scale');
    $(".chat-box").toggle('scale');
    
  })
  
})
