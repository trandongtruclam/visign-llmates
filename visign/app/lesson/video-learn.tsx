"use client";

import { useState, useEffect } from "react";

import { Check } from "lucide-react";

import { Button } from "@/components/ui/button";

type VideoLearnProps = {
  videoUrl: string;
  question: string;
  onContinue: () => void;
};

export const VideoLearn = ({
  videoUrl,
  question,
  onContinue,
}: VideoLearnProps) => {
  const [watched, setWatched] = useState(false);

  // Auto-enable continue button after 3 seconds
  useEffect(() => {
    const timer = setTimeout(() => {
      setWatched(true);
    }, 3000);

    return () => clearTimeout(timer);
  }, []);

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-col items-center gap-y-8">
      <div className="text-center">
        <h2 className="mb-2 text-2xl font-bold text-neutral-700">
          Học dấu hiệu mới
        </h2>
        <p className="text-lg text-muted-foreground">
          Xem video và ghi nhớ dấu hiệu này
        </p>
      </div>

      {/* Video Display */}
      <div className="w-full max-w-[600px]">
        <div className="relative aspect-video w-full overflow-hidden rounded-xl border-4 border-blue-500 bg-gray-100 shadow-xl">
          <iframe
            src={`${videoUrl}?autoplay=1&loop=1&title=0&byline=0&portrait=0&badge=0`}
            className="h-full w-full"
            frameBorder="0"
            allow="autoplay; fullscreen; picture-in-picture"
            allowFullScreen
          />
        </div>

        {/* Sign Label */}
        <div className="mt-6 text-center">
          <div className="inline-block rounded-2xl border-2 border-blue-500 bg-blue-100 px-8 py-3">
            <p className="text-2xl font-bold text-blue-700">{question}</p>
          </div>
        </div>
      </div>

      {/* Continue Button */}
      <Button
        onClick={onContinue}
        size="lg"
        variant="super"
        className="min-w-[200px]"
        disabled={!watched}
      >
        <Check className="mr-2" />
        {watched ? "Tiếp tục" : "Đang xem... (3s)"}
      </Button>

      {!watched && (
        <p className="text-sm text-muted-foreground">
          Xem video ít nhất 3 giây để tiếp tục
        </p>
      )}
    </div>
  );
};
