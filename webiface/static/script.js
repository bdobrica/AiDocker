({
    q:function(i){return document.querySelectorAll(i);},
    i:function(){
        this.q('.send')[0].addEventListener('click',function(e){
            e.preventDefault();
            var t=this.q('.text-box textarea')[0].value;
            this.s(t);
        }.bind(this));
    },
    a:function(t){
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
    },
    s:function(t){alert("send message:"+t);this.a(t);}
}).i();
