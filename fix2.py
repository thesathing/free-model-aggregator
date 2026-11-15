from pathlib import Path
p=Path('/home/ubuntu/free-model-aggregator-work/entry/main.ts')
s=p.read_text()
s=s.replace("function gemJson(raw:Json,model:string,api:'openai'|'anthropic'):Json{const c=", "function gemText(v:unknown):string{if(!Array.isArray(v))return '';return v.flatMap(x=>isRecord(x)&&typeof x.text==='string'?[x.text]:[]).join('');}\nfunction gemJson(raw:Json,model:string,api:'openai'|'anthropic'):Json{const c=")
s=s.replace("const text=textOf(content.parts);const u=", "const text=gemText(content.parts);const u=",1)
s=s.replace("text+=textOf(co.parts);const u=", "text+=gemText(co.parts);const u=",1)
s=s.replace("function mark(p:string){const n=(used.get(p)??0)+1;used.set(p,n);if(n>=(reserve[p]??99))spent.add(p);}", "function mark(p:string){const n=(used.get(p)??0)+1;used.set(p,n);}")
old="url=`${base}/v1/chat/completions`;out=api==='openai'?{...b,model:route.model.id}:anthropicToOpenAI(b,route.model.id);if(b.stream===true)headers.accept='text/event-stream';"
new="url=`${base}/v1/chat/completions`;out=api==='openai'?{...b,model:route.model.id}:anthropicToOpenAI(b,route.model.id);if(b.stream===true){headers.accept='text/event-stream';out.stream_options={...(isRecord(out.stream_options)?out.stream_options:{}),include_usage:true};}"
assert old in s
s=s.replace(old,new)
p.write_text(s)
