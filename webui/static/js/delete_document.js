({
    q:function(i){return document.querySelectorAll(i);},
    i:function(){
        this.q('.action__delete')[0].addEventListener('click',function(e){
            e.preventDefault();
            var d=new FormData();
            this.r('/api/document/' + this.q('[name="document_id"]')[0].value, null).then(function(r){
                console.log(r);
                this.k=r.token;
            }.bind(this));
        }.bind(this));
    },
    r:async function(u,d){
        var r=await fetch(u,{
            method:'DELETE',
            body:d
        });
        var j=await r.json();
        return j;
    },
}).i();
