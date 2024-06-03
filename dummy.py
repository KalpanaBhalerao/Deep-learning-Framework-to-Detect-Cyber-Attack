from fastapi import FastAPI

#https://youtu.be/52c7Kxp_14E?si=FJ3H_VY2EfeklBjV

app = FastAPI()


@app.post("/geMyName")
async def geMyName():
    return {"name":"Pratiksha"}

@app.Get("/getMethod")
async def getMethod():
    return {"name":"This is get method"}
# HTTP Verbs / method types
#put , patch , delet method
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=4200)
