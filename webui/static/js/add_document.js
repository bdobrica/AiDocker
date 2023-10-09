({
    q:function(i){return document.querySelectorAll(i);},
    i:function(){
        this.s=this.q('select[name="search_space"]')[0];
        this.t=this.q('input[name="add_search_space"]')[0];

        this.s.addEventListener('change',function(e){
            e.preventDefault();
            if (this.s.options[this.s.selectedIndex].value=='add') {
                this.s.style.display='none';
                this.t.style.display='inline-block';
            }
        }.bind(this));
        this.t.addEventListener('change',function(e){
            e.preventDefault();
            if (this.t.value) {
                this.s.style.display='inline-block';
                this.t.style.display='none';
                this.s.add(new Option(this.t.value,this.t.value));
            }
        }.bind(this));
        this.t.addEventListener("keyup", function(e){
            console.log(e.keyCode);
            if (e.keyCode === 13) {
                e.preventDefault();
                this.t.dispatchEvent(new Event('change'));
                return;
            }
            if (e.keyCode === 27) {
                e.preventDefault();
                this.s.style.display='inline-block';
                this.t.style.display='none';
                this.t.value='';
                return;
            }
        }.bind(this));
        this.q('button.send')[0].addEventListener('click',function(e){
            e.preventDefault();
            var d=new FormData();
            d.append('document',this.q('[name="document"]')[0].files[0]);
            d.append("search_space",this.s.options[this.s.selectedIndex].value);    
            this.r('/api/document', d).then(function(r){
                console.log(r);
                this.k=r.token;
            }.bind(this));
        }.bind(this));
    },
    r:async function(u,d){
        var r=await fetch(u,{
            method:'PUT',
            body:d
        });
        var j=await r.json();
        return j;
    },
}).i();
