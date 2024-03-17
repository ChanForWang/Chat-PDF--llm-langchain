css = '''
<style>
.chat-container{
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}
.chat-message {
    padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e;
    margin-left: auto;
    align-items:top;
}
.chat-message.bot {
    background-color: #475063;
    margin-right: auto;
    align-items:top;
}
.chat-message .avatar {
  width: 3%;
}
.chat-message .avatar img {
  max-width: 45px;
  max-height: 45px;
  border-radius: 50%;
  object-fit: cover;
  padding-top:6px;
}
.chat-message .message {
  width: 80%;
  padding: 0.5rem 0.5rem 0.5rem 2.5rem;
  color: #fff;
  display:flex;
  justify-content:center;
  align-items:center;
}
.chat-message.user .message{
  color: #87cefa
}

.stButton Button{
  background:#ff0000;
}
.stButton Button p{
  font-weight:bold;
}
.stTextInput{
  position:fixed;
  Bottom: 35px;
  z-index:9999;
}

*, ::before, ::after{
  box-sizing:unset;
}
'''

bot_template = '''
<div class=chat-container>
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://i.ibb.co/4jZ4YtH/chatbot.jpg">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
</div>
    '''

user_template = '''
<div class=chat-container>
    <div class="chat-message user">
        <div class="avatar">
            <img src="https://i.ibb.co/QYNK1tH/bg-kobe9.jpg">
        </div>    
        <div class="message">{{MSG}}</div>
    </div>
</div>
    '''


