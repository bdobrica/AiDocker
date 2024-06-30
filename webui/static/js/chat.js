({
    q:function(i){return document.querySelectorAll(i);},
    i:function(){
        this.q('.send')[0].addEventListener('click',function(e){
            e.preventDefault();
            var t=this.q('.chat__text__box textarea')[0];
            this.s(t.value);
            t.value='';
        }.bind(this));
    },
    a:function(t)   {
        var w=this.q(".messages")[0];
        var ms=this.q(".messages .message");
        var m=ms[ms.length-1].cloneNode(true);
        m.querySelector(".message__bubble").innerHTML=t;
        if (m.classList.contains("assistant")) {
            m.classList.remove("assistant");
            m.classList.add("user");
        }
        else {
            m.classList.remove("user");
            m.classList.add("assistant");
        }
        w.appendChild(m);
        w.scrollTop=w.scrollHeight;
    },
    r:async function(u,d){
        var r=await fetch(u,{
            method:'POST',
            headers:{
                'Content-Type':'application/json'
            },
            body:JSON.stringify(d)
        });
        var j=await r.json();
        return j;
    },
    l:function(e){
        var l=this.q('.chat__loading')[0];
        if(e)l.style.display='block';else l.style.display='none';
    },
    s:function(t){
        this.a(t);
        this.l(true);
        this.r('/api/chat',{prompt:t}).then(function(r){
            this.a(r.answer.content);
            this.l(false);
        }.bind(this));
    }
}).i();