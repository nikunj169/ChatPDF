css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    align: right,
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.citations {
    margin-top: 0.75rem;
    padding: 0.5rem 0.75rem;
    background-color: #3a4256;
    border-radius: 0.35rem;
    font-size: 0.85rem;
    color: #b0b8c8;
}
.citations strong {
    color: #8a9bba;
}
.citation-item {
    padding: 0.25rem 0;
    border-bottom: 1px solid #4a5368;
}
.citation-item:last-child {
    border-bottom: none;
}
.eval-metric {
    padding: 0.5rem 1rem;
    background-color: #2b313e;
    border-radius: 0.35rem;
    margin-bottom: 0.5rem;
}
</style>
'''
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/6134/6134346.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message" style="text-align:right">{{MSG}}</div>
    <div class="avatar">
        <img src="https://png.pngtree.com/png-vector/20190321/ourmid/pngtree-vector-users-icon-png-image_856952.jpg">
    </div>
</div>
'''

citation_template = '''
<div class="citations">
    <strong>Sources:</strong>
    {{CITATIONS}}
</div>
'''
