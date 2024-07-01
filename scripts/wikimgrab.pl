#!/usr/bin/perl
 
use strict;
use warnings;

use URI::Escape;
use Digest::MD5 qw(md5_hex);
use LWP::UserAgent;

my $ua = LWP::UserAgent->new;
$ua->timeout(15);
$ua->env_proxy;
$ua->show_progress(1);
 
foreach my $image( @ARGV ) {
       $image = uri_unescape($image);
 
       $image =~ s/ /_/g;
 
        $image =~ s/^(File|Image)://ig;
 
        $image =~ s/^(\w)/uc($1)/e;
 
        my $digest = lc(md5_hex( $image ));
        my $a = substr $digest, 0, 1;
        my $b = substr $digest, 0, 2;
        my $path = "http://upload.wikimedia.org/wikipedia/commons/$a/$b/$image";
        if ($ua->mirror( $path, $image )->is_error) { #if failed, look for redirects
        warn("Could not get image directly - looking for alternative name on main image page");
        my $basepage = "http://commons.wikimedia.org/wiki/File:$image";
        my $response = $ua->get($basepage);
        if ($response->content =~ m!<link rel="canonical" href="/wiki/(.+?)"!) {
            $image = uri_unescape($1); #found an alternative "canonical" link
        } else {
            $image = uri_unescape($response->filename); #this is a redirect 
        }
        $image =~ s/ /_/g;

        $image =~ s/^(File|Image)://ig;

        $image =~ s/^(\w)/uc($1)/e;

        $digest = lc(md5_hex( $image ));
        $a = substr $digest, 0, 1;
        $b = substr $digest, 0, 2;
        $path = "http://upload.wikimedia.org/wikipedia/commons/$a/$b/$image";
        $ua->mirror( $path, $image );
        }
}

